from llm_gpu_fit.core.types import GPU, Framework, MemoryFit, Model, Quantization


_BYTES_PER_PARAM: dict[str, float] = {
    "fp16": 2.0, "bf16": 2.0,
    "fp8": 1.0, "int8": 1.0,
    "int4": 0.5, "awq": 0.5, "gptq": 0.5,
    "q4_k_m": 0.55, "q5_k_m": 0.65, "q8_0": 1.05,
}

_FRAMEWORK_OVERHEAD_GB: dict[str, float] = {
    "vllm": 1.5, "sglang": 1.5, "tgi": 1.5,
    "llama.cpp": 0.5, "mlx": 0.5,
    "tensorrt-llm": 2.0,
}


def bytes_per_param(quant: Quantization) -> float:
    if quant not in _BYTES_PER_PARAM:
        raise ValueError(f"Unknown quantization: {quant}")
    return _BYTES_PER_PARAM[quant]


def weights_gb(model: Model, quant: Quantization) -> float:
    return model.params_total_b * bytes_per_param(quant) * 1.05


def kv_cache_gb(model: Model, batch: int, seq_len: int,
                kv_quant: Quantization = "fp16") -> float:
    head_dim = model.hidden_dim / model.num_attention_heads
    bytes_per_token = (
        batch * seq_len * model.num_layers * 2
        * model.num_kv_heads * head_dim
        * bytes_per_param(kv_quant)
    )
    return bytes_per_token / 1e9


def activations_gb(weights: float) -> float:
    return weights * 0.07


def framework_overhead_gb(framework: Framework, gpu_count: int) -> float:
    base = _FRAMEWORK_OVERHEAD_GB.get(framework, 1.5)
    return base * gpu_count


def compute_memory_fit(
    model: Model,
    gpus: list[GPU],
    gpu_count: int,
    quantization: Quantization,
    framework: Framework,
    context_target: int,
    concurrency: int,
    kv_quant: Quantization = "fp16",
) -> MemoryFit:
    """gpus[0]를 gpu_count배로 사용하는 단순 모델."""
    gpu = gpus[0]
    total_available = gpu.total_vram(gpu_count) * (
        0.75 if gpu.form_factor == "apple_silicon" else 1.0
    )

    w = weights_gb(model, quantization)
    fw = framework_overhead_gb(framework, gpu_count)
    kv = kv_cache_gb(model, batch=concurrency, seq_len=context_target, kv_quant=kv_quant)
    act = activations_gb(w)

    total_used = w + fw + kv + act
    headroom_for_kv = total_available * 0.95 - w - fw - act

    fits = total_used <= total_available * 0.95
    util = total_used / total_available if total_available > 0 else 1.0
    if not fits:
        status = "no"
    elif util >= 0.85:
        status = "tight"
    else:
        status = "ok"

    max_ctx = 0
    if w + fw + act < total_available * 0.95:
        lo, hi = 0, model.context_window
        while lo < hi:
            mid = (lo + hi + 1) // 2
            test_kv = kv_cache_gb(model, batch=concurrency, seq_len=mid, kv_quant=kv_quant)
            if w + fw + act + test_kv <= total_available * 0.95:
                lo = mid
            else:
                hi = mid - 1
        max_ctx = lo

    kv_per_req = kv_cache_gb(model, batch=1, seq_len=context_target, kv_quant=kv_quant)
    max_conc = int(headroom_for_kv / kv_per_req) if kv_per_req > 0 else 0
    max_conc = max(0, max_conc)

    return MemoryFit(
        fits=fits,
        fit_status=status,
        weights_gb=w,
        kv_cache_gb=kv,
        framework_overhead_gb=fw,
        total_used_gb=total_used,
        total_available_gb=total_available,
        max_context_supported=max_ctx,
        max_concurrency_at_target_ctx=max_conc,
    )
