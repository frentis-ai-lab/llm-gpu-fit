from llm_gpu_fit.core.data import load_benchmarks_for, load_models
from llm_gpu_fit.core.memory import compute_memory_fit
from llm_gpu_fit.core.topology import check_topology
from llm_gpu_fit.core.types import (
    Framework,
    GPU,
    Model,
    Quantization,
    Recommendation,
    UserInput,
)
from llm_gpu_fit.core.usecases import UseCase, score_quality


_QUANT_PREFERENCE: list[Quantization] = [
    "bf16", "fp16", "fp8", "int8", "awq", "gptq", "int4",
    "q8_0", "q5_k_m", "q4_k_m",
]


def is_capability_satisfied(required: list[str], model_caps: list[str],
                            model_modalities: list[str] | None = None) -> bool:
    """required는 modality나 capability 어느 쪽이든 만족하면 OK."""
    pool = list(model_caps) + list(model_modalities or [])
    return all(c in pool for c in required)


def pick_quantization(model: Model, gpus: list[GPU], gpu_count: int,
                      framework: Framework, context_target: int,
                      concurrency: int,
                      preference: Quantization | None = None) -> Quantization | None:
    if preference:
        fit = compute_memory_fit(model, gpus, gpu_count, preference, framework,
                                 context_target, concurrency)
        if fit.fits:
            return preference

    for quant in _QUANT_PREFERENCE:
        fit = compute_memory_fit(model, gpus, gpu_count, quant, framework,
                                 context_target, concurrency)
        if fit.fits:
            return quant
    return None


def _constraint_score(ui: UserInput, model: Model) -> float:
    checks = []
    if ui.commercial_required:
        checks.append(model.license_commercial_ok)
    if ui.korean_priority:
        checks.append("ko_native" in model.capabilities)
    if ui.tool_calling_required:
        checks.append("tool_use" in model.capabilities)
    if ui.onprem_required:
        checks.append(True)
    return 1.0 if not checks else sum(checks) / len(checks)


def _why(model: Model, ui: UserInput, quant: Quantization, fit, topo,
         quality: float, bench: dict[str, float], use_case: UseCase) -> list[str]:
    items: list[str] = []
    weighted = sorted(
        ((b, bench[b], use_case.benchmark_weights.get(b, 0))
         for b in bench if b in use_case.benchmark_weights),
        key=lambda x: x[1] * x[2], reverse=True,
    )[:3]
    for bid, score, _w in weighted:
        items.append(f"{bid.replace('_', ' ').upper()} {score:.1f}점")

    if model.license_commercial_ok:
        items.append(f"{model.license} → 상용 가능")
    if "ko_native" in model.capabilities:
        items.append("한국어 네이티브")
    if quant in ("bf16", "fp16"):
        items.append(f"양자화 없이 {quant.upper()}로 동작 → 품질 손실 최소")
    elif quant in ("int4", "awq", "gptq"):
        items.append(f"{quant.upper()} 양자화로 메모리 절감")
    if topo.recommended_tp == 1 and ui.gpu_count == 1:
        items.append("단일 GPU → 토폴로지 단순, NVLink 불필요")
    if "tool_use" in model.capabilities and ui.tool_calling_required:
        items.append("Tool calling 지원")
    return items


def _tradeoffs(model: Model, ui: UserInput, fit, topo) -> list[str]:
    items: list[str] = list(topo.warnings)
    if fit.fit_status == "tight":
        items.append(
            f"⚠ 메모리 빠듯 (사용 {fit.total_used_gb:.0f}/{fit.total_available_gb:.0f}GB)"
        )
    if model.is_moe():
        items.append(
            f"MoE 모델 ({model.params_active_b:.0f}B active / "
            f"{model.params_total_b:.0f}B total) — 가중치는 전체 로드 필요"
        )
    if ui.commercial_required and not model.license_commercial_ok:
        items.append("❌ 라이선스가 상용 사용을 허용하지 않음")
    return items


def recommend(ui: UserInput, gpus_by_id: dict[str, GPU],
              use_cases_by_id: dict[str, UseCase],
              top_k: int = 3) -> list[Recommendation]:
    if ui.use_case not in use_cases_by_id:
        raise ValueError(f"Unknown use case: {ui.use_case}")
    if ui.gpu_id not in gpus_by_id:
        raise ValueError(f"Unknown GPU: {ui.gpu_id}")

    use_case = use_cases_by_id[ui.use_case]
    gpu = gpus_by_id[ui.gpu_id]
    framework: Framework = "mlx" if gpu.form_factor == "apple_silicon" else "vllm"

    candidates: list[Recommendation] = []
    for model in load_models():
        if not is_capability_satisfied(use_case.required_capabilities,
                                       model.capabilities, model.modalities):
            continue
        if ui.commercial_required and not model.license_commercial_ok:
            continue

        quant = pick_quantization(
            model, [gpu], ui.gpu_count, framework, ui.context_target,
            ui.concurrency, preference=ui.quantization_preference,
        )
        if quant is None:
            continue

        fit = compute_memory_fit(model, [gpu], ui.gpu_count, quant, framework,
                                 ui.context_target, ui.concurrency)
        if not fit.fits:
            continue

        topo = check_topology(model, gpu, ui.gpu_count)
        if not topo.tp_compatible:
            continue

        bench = load_benchmarks_for(model.id)
        quality = score_quality(bench, use_case.benchmark_weights)
        constraint = _constraint_score(ui, model)

        pref_bonus = 0.0
        for cap in use_case.preferred_capabilities:
            if cap in model.capabilities:
                pref_bonus += 0.05

        raw = 0.65 * quality + 0.35 * constraint * 100
        raw *= (1 - topo.nvlink_penalty)
        raw *= (1 + pref_bonus)
        if fit.fit_status == "tight":
            raw *= 0.7

        rec = Recommendation(
            model=model, quantization=quant, framework=framework,
            memory=fit, topology=topo,
            quality_score=quality, constraint_score=constraint * 100,
            final_score=raw,
            why=_why(model, ui, quant, fit, topo, quality, bench, use_case),
            tradeoffs=_tradeoffs(model, ui, fit, topo),
        )
        candidates.append(rec)

    candidates.sort(key=lambda r: r.final_score, reverse=True)
    return candidates[:top_k]


def suggest_smaller_alternative(top_rec: Recommendation, ui: UserInput,
                                gpus_by_id: dict[str, GPU],
                                use_cases_by_id: dict[str, UseCase]
                                ) -> Recommendation | None:
    if ui.gpu_count <= 1:
        return None
    alt_ui = UserInput(
        use_case=ui.use_case, gpu_id=ui.gpu_id, gpu_count=1,
        commercial_required=ui.commercial_required,
        korean_priority=ui.korean_priority,
        onprem_required=ui.onprem_required,
        tool_calling_required=ui.tool_calling_required,
        concurrency=ui.concurrency, context_target=ui.context_target,
    )
    alts = recommend(alt_ui, gpus_by_id, use_cases_by_id, top_k=1)
    if not alts:
        return None
    alt = alts[0]
    if top_rec.quality_score > 0 and alt.quality_score < top_rec.quality_score * 0.7:
        return None
    return alt
