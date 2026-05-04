from llm_gpu_fit.core.types import GPU, Recommendation


def _cost_section(gpu: GPU, count: int) -> str:
    cloud = gpu.monthly_cloud_cost(count)
    owned = gpu.amortized_monthly_owned(count) if gpu.msrp_usd > 0 else 0
    parts = []
    if cloud > 0:
        parts.append(f"클라우드 24/7 운용 시 **월 ${cloud:,.0f}**")
    if owned > 0:
        parts.append(f"자체 보유 시 **월 ${owned:,.0f}** (MSRP ÷ 36개월)")
    if not parts:
        return ""
    return "**💰 추정 비용**: " + " · ".join(parts) + "\n"


def render_recommendation_card(rec: Recommendation, rank: int,
                               gpu: GPU | None = None,
                               gpu_count: int = 1,
                               summary_mode: str = "engineer") -> str:
    """summary_mode: 'engineer' (상세) | 'executive' (임원용)."""
    medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, str(rank))

    if summary_mode == "executive":
        return _render_executive(rec, rank, medal, gpu, gpu_count)

    fit_emoji = {"ok": "✅", "tight": "⚠", "no": "❌"}[rec.memory.fit_status]
    why_md = "\n".join(f"- {w}" for w in rec.why) if rec.why else "- (벤치마크 데이터 부족)"
    tradeoffs_md = "\n".join(f"- {t}" for t in rec.tradeoffs) if rec.tradeoffs else "- 없음"

    cost_section = _cost_section(gpu, gpu_count) if gpu else ""
    company = rec.model.company_or_family

    quant_note = _quant_reason(rec)

    return f"""
### {medal} {rank}순위: {rec.model.display_name}
*{company} · 출시 {rec.model.release_date or "—"}*

**구성**: {rec.quantization.upper()} · {rec.framework} · 컨텍스트 {rec.memory.max_context_supported:,} 토큰까지

{quant_note}

**메모리**: {fit_emoji} {rec.memory.total_used_gb:.1f}GB / {rec.memory.total_available_gb:.1f}GB 사용
(weights {rec.memory.weights_gb:.1f}GB · KV {rec.memory.kv_cache_gb:.1f}GB · framework {rec.memory.framework_overhead_gb:.1f}GB)

**최대 동시성**: 목표 컨텍스트 기준 {rec.memory.max_concurrency_at_target_ctx} 요청

{cost_section}
**점수**: 품질 {rec.quality_score:.1f}/100 · 제약 충족 {rec.constraint_score:.0f}% · **종합 {rec.final_score:.1f}**

#### ✅ 왜 이 모델인가
{why_md}

#### ⚠ 알아둘 것
{tradeoffs_md}
"""


_QUANT_NOTES: dict[str, str] = {
    "bf16": "🔍 **BF16 선택 이유**: GPU에 양자화 없이 들어감 → 품질 손실 0%, 가장 안전",
    "fp16": "🔍 **FP16 선택 이유**: 양자화 없이 들어감, 품질 손실 0%",
    "fp8": "🔍 **FP8 선택 이유**: H100/H200 네이티브 지원, 품질 손실 1-2% (BF16 대비 메모리 절반)",
    "int8": "🔍 **INT8 선택 이유**: BF16 안 들어감 → 품질 손실 2-3%로 메모리 절반",
    "awq": "🔍 **AWQ 선택 이유**: INT4 기반 권장 양자화, 메모리 4배 절감 / 품질 손실 5-10%",
    "gptq": "🔍 **GPTQ 선택 이유**: AWQ 대안 INT4, 메모리 4배 절감 / 품질 손실 5-10%",
    "int4": "🔍 **INT4 선택 이유**: 메모리 4배 절감 / 품질 손실 5-10%",
    "q4_k_m": "🔍 **Q4_K_M (GGUF) 선택 이유**: Ollama/llama.cpp INT4 표준",
    "q5_k_m": "🔍 **Q5_K_M (GGUF) 선택 이유**: Q4보다 약간 큰 GGUF, 품질 1-2% ↑",
    "q8_0": "🔍 **Q8_0 (GGUF) 선택 이유**: Ollama/llama.cpp INT8, 품질 손실 거의 없음",
}


def _quant_reason(rec: Recommendation) -> str:
    """선택된 양자화의 사유를 한 줄로 설명."""
    q = rec.quantization
    base = _QUANT_NOTES.get(q, "")
    # 사용자가 INT4를 선택했지만 더 좋은 것을 자동 선택한 경우 안내
    weights_gb = rec.memory.weights_gb
    return f"{base}\n메모리 점유 = weights {weights_gb:.1f}GB"


def _render_executive(rec: Recommendation, rank: int, medal: str,
                      gpu: GPU | None, gpu_count: int) -> str:
    """임원용: 비즈니스 언어 1-2 문단."""
    company = rec.model.company_or_family
    license_friendly = (
        "✓ 회사에서 자유롭게 사용 가능 (별도 라이선스 비용 없음)"
        if rec.model.license_commercial_ok
        else "✗ 연구·내부 평가 한정 (제품 배포 시 별도 협의 필요)"
    )

    cost_line = ""
    if gpu:
        cloud = gpu.monthly_cloud_cost(gpu_count)
        owned = gpu.amortized_monthly_owned(gpu_count)
        if cloud > 0 or owned > 0:
            cloud_str = f"월 약 **${cloud:,.0f}**" if cloud > 0 else "—"
            owned_str = f"월 약 **${owned:,.0f}**" if owned > 0 else "—"
            cost_line = f"\n💰 **인프라 비용**: 클라우드 {cloud_str} · 자체 보유 {owned_str}"

    quality_label = (
        "최상위" if rec.quality_score >= 70
        else "상위" if rec.quality_score >= 50
        else "중간" if rec.quality_score >= 30
        else "보통"
    )
    fit_label = {
        "ok": "충분히 들어감", "tight": "빠듯하지만 가능",
        "no": "메모리 초과",
    }[rec.memory.fit_status]

    return f"""
### {medal} {rank}순위: {rec.model.display_name}
**제공사**: {company} · **출시**: {rec.model.release_date or "최근"} · **라이선스**: {license_friendly}

이 GPU 구성에 **{fit_label}** ({rec.memory.total_used_gb:.0f}GB / {rec.memory.total_available_gb:.0f}GB).
요청 용도 기준 품질은 **{quality_label}** 수준입니다 (100점 만점 중 {rec.quality_score:.0f}점).
{cost_line}

**핵심 강점**: {rec.why[0] if rec.why else "벤치마크 데이터 보강 중"}

---
"""


def render_results(recs: list[Recommendation],
                   gpu: GPU | None = None, gpu_count: int = 1,
                   summary_mode: str = "engineer") -> str:
    if not recs:
        return ("### 😅 추천할 모델이 없습니다\n\n"
                "이 GPU 구성에 들어가는 모델이 없거나 모든 제약을 만족하는 모델이 없습니다. "
                "GPU 수를 늘리거나 제약 체크를 줄여보세요.")
    cards = [render_recommendation_card(r, i + 1, gpu, gpu_count, summary_mode)
             for i, r in enumerate(recs)]
    return "\n".join(cards)


def render_alternative_section(top: Recommendation,
                               alt: Recommendation | None) -> str:
    if alt is None:
        return ""
    delta = alt.quality_score - top.quality_score
    delta_str = f"{delta:+.1f}점"
    return f"""
### 💡 대안: 1장 구성으로 더 단순하게

같은 GPU **1장**에서 가능한 추천: **{alt.model.display_name}** ({alt.quantization.upper()})

| 항목 | 다중 GPU 추천 | 1장 대안 |
|------|------------|--------|
| 모델 | {top.model.display_name} | {alt.model.display_name} |
| 양자화 | {top.quantization.upper()} | {alt.quantization.upper()} |
| 품질 점수 | {top.quality_score:.1f} | {alt.quality_score:.1f} ({delta_str}) |
| 메모리 사용 | {top.memory.total_used_gb:.0f}GB / {top.memory.total_available_gb:.0f}GB | {alt.memory.total_used_gb:.0f}GB / {alt.memory.total_available_gb:.0f}GB |
| 토폴로지 페널티 | {top.topology.nvlink_penalty*100:.0f}% | 0% |

품질 차이가 작고 운영 단순성·통신 오버헤드를 고려하면 1장 구성이 나을 수 있습니다.

---
"""


def render_plain_text(recs: list[Recommendation]) -> str:
    """Slack/이메일 붙여넣기용 단순 텍스트."""
    if not recs:
        return "추천할 모델이 없습니다."
    lines = ["📊 LLM-GPU-Fit 추천 결과\n"]
    for i, r in enumerate(recs, 1):
        medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(i, f"{i}.")
        lines.append(
            f"{medal} {r.model.display_name} ({r.model.company_or_family})\n"
            f"   · 양자화 {r.quantization.upper()} / {r.framework}\n"
            f"   · 메모리 {r.memory.total_used_gb:.0f}/{r.memory.total_available_gb:.0f}GB\n"
            f"   · 품질 {r.quality_score:.0f}/100, 종합 {r.final_score:.0f}\n"
            f"   · {r.model.hf_repo}\n"
        )
    lines.append("\n— https://huggingface.co/spaces/frentis/llm-gpu-fit")
    return "\n".join(lines)
