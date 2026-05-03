"""스펙 §10 회귀 케이스. 메모리 계산이 발표값/직관과 어긋나지 않는지 확인."""
import pytest

from llm_gpu_fit.core.data import load_gpus, load_models
from llm_gpu_fit.core.memory import compute_memory_fit


@pytest.fixture
def gpus():
    return {g.id: g for g in load_gpus()}


@pytest.fixture
def models():
    return {m.id: m for m in load_models()}


def test_llama_70b_bf16_does_not_fit_h100_single(gpus, models):
    fit = compute_memory_fit(
        models["llama_3_3_70b"], [gpus["h100_80"]], gpu_count=1,
        quantization="bf16", framework="vllm",
        context_target=4096, concurrency=1,
    )
    assert fit.fits is False
    assert fit.total_used_gb > 80


def test_llama_70b_int4_fits_h100_single(gpus, models):
    fit = compute_memory_fit(
        models["llama_3_3_70b"], [gpus["h100_80"]], gpu_count=1,
        quantization="int4", framework="vllm",
        context_target=8192, concurrency=1,
    )
    assert fit.fits is True


def test_solar_pro_22b_bf16_does_not_fit_rtx4090(gpus, models):
    fit = compute_memory_fit(
        models["solar_pro_22b"], [gpus["rtx4090"]], gpu_count=1,
        quantization="bf16", framework="vllm",
        context_target=4096, concurrency=1,
    )
    assert fit.fits is False


def test_qwen3_32b_int4_fits_h100(gpus, models):
    fit = compute_memory_fit(
        models["qwen3_32b"], [gpus["h100_80"]], gpu_count=1,
        quantization="int4", framework="vllm",
        context_target=32768, concurrency=4,
    )
    assert fit.fits is True


def test_long_context_reduces_max_concurrency(gpus, models):
    fit_8k = compute_memory_fit(
        models["qwen3_32b"], [gpus["h100_80"]], gpu_count=1,
        quantization="int4", framework="vllm",
        context_target=8192, concurrency=1,
    )
    fit_128k = compute_memory_fit(
        models["qwen3_32b"], [gpus["h100_80"]], gpu_count=1,
        quantization="int4", framework="vllm",
        context_target=131072, concurrency=1,
    )
    assert fit_8k.max_concurrency_at_target_ctx > fit_128k.max_concurrency_at_target_ctx


def test_4090x4_fits_70b_int4(gpus, models):
    fit = compute_memory_fit(
        models["llama_3_3_70b"], [gpus["rtx4090"]], gpu_count=4,
        quantization="int4", framework="vllm",
        context_target=8192, concurrency=1,
    )
    assert fit.fits is True


def test_apple_silicon_uses_75pct_unified_memory(gpus, models):
    fit = compute_memory_fit(
        models["qwen3_32b"], [gpus["m4_max_128"]], gpu_count=1,
        quantization="bf16", framework="mlx",
        context_target=8192, concurrency=1,
    )
    # M4 Max 128GB unified → 96GB vram_gb → 0.75 적용 = 72GB available
    assert fit.total_available_gb == pytest.approx(72, rel=0.05)
