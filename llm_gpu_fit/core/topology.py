from llm_gpu_fit.core.types import GPU, Model, TopologyCheck


def recommended_tp_size(weights_gb: float, single_gpu_vram: float,
                        available_gpu_count: int, has_nvlink: bool) -> int:
    if weights_gb <= single_gpu_vram * 0.6:
        return 1
    needed = max(1, -(-int(weights_gb * 1.4) // max(1, int(single_gpu_vram))))
    needed = min(needed, available_gpu_count)
    for candidate in [1, 2, 4, 8, 16]:
        if candidate >= needed and candidate <= available_gpu_count:
            return candidate
    return available_gpu_count


def check_topology(model: Model, gpu: GPU, gpu_count: int) -> TopologyCheck:
    warnings: list[str] = []

    tp_compatible = True
    tp_reason = ""
    if gpu_count > 1:
        if model.num_attention_heads % gpu_count != 0:
            tp_compatible = False
            tp_reason = (f"모델의 attention heads({model.num_attention_heads})가 "
                         f"GPU 수({gpu_count})로 나누어 떨어지지 않음")
        elif model.num_kv_heads % gpu_count != 0:
            tp_compatible = False
            tp_reason = (f"모델의 KV heads({model.num_kv_heads})가 "
                         f"GPU 수({gpu_count})로 나누어 떨어지지 않음")

    nvlink_penalty = 0.0
    if gpu_count > 1 and not gpu.nvlink and gpu.form_factor in ("consumer", "workstation"):
        size = model.params_total_b
        if size >= 70:
            nvlink_penalty = 0.25
            warnings.append(
                f"⚠ {gpu.name} {gpu_count}장은 NVLink 없음 → "
                f"PCIe all-reduce, 70B+ 모델에서 TPS 약 25% 저하 추정"
            )
        elif size >= 30:
            nvlink_penalty = 0.15
            warnings.append(
                f"⚠ {gpu.name} {gpu_count}장은 NVLink 없음 → "
                f"PCIe 통신, 30B+ 모델에서 TPS 약 15% 저하 추정"
            )
        else:
            warnings.append(
                f"⚠ {gpu.name} {gpu_count}장은 NVLink 없음 → "
                f"PCIe 통신, 작은 모델은 영향 미미"
            )

    from llm_gpu_fit.core.memory import weights_gb as _w
    weights = _w(model, "bf16")
    rec_tp = recommended_tp_size(weights_gb=weights,
                                 single_gpu_vram=gpu.vram_gb,
                                 available_gpu_count=gpu_count,
                                 has_nvlink=gpu.nvlink)

    if rec_tp < gpu_count and tp_compatible:
        warnings.append(
            f"💡 이 모델은 TP={rec_tp}이 최적입니다 (TP={gpu_count}는 통신 오버헤드 추가)"
        )

    return TopologyCheck(
        tp_compatible=tp_compatible,
        tp_reason=tp_reason,
        nvlink_penalty=nvlink_penalty,
        recommended_tp=rec_tp,
        warnings=warnings,
    )
