import pytest

from llm_gpu_fit.core.memory import (
    bytes_per_param,
    compute_memory_fit,
    kv_cache_gb,
    weights_gb,
)
from llm_gpu_fit.core.types import GPU, Model


@pytest.fixture
def llama_70b():
    return Model(
        id="llama_3_3_70b", display_name="Llama 3.3 70B", family="llama",
        params_total_b=70.6, params_active_b=70.6, context_window=131072,
        num_attention_heads=64, num_kv_heads=8, num_layers=80, hidden_dim=8192,
        modalities=["text"], capabilities=["tool_use"], license="Llama 3.3 Community",
        license_commercial_ok=True, hf_repo="meta-llama/Llama-3.3-70B-Instruct",
    )


@pytest.fixture
def h100():
    return GPU(id="h100_80", name="H100 80GB", vendor="nvidia",
               vram_gb=80, mem_bandwidth_gbs=3350, fp16_tflops=989,
               int8_tops=1979, nvlink=True, form_factor="datacenter")


@pytest.fixture
def rtx4090():
    return GPU(id="rtx4090", name="RTX 4090", vendor="nvidia",
               vram_gb=24, mem_bandwidth_gbs=1008, fp16_tflops=82.6,
               int8_tops=660, nvlink=False, form_factor="consumer")


def test_bytes_per_param_known_quantizations():
    assert bytes_per_param("fp16") == 2.0
    assert bytes_per_param("bf16") == 2.0
    assert bytes_per_param("int4") == 0.5
    assert bytes_per_param("awq") == 0.5
    assert bytes_per_param("q4_k_m") == 0.55


def test_weights_gb_llama_70b_bf16(llama_70b):
    w = weights_gb(llama_70b, "bf16")
    assert 145 < w < 152


def test_weights_gb_llama_70b_int4(llama_70b):
    w = weights_gb(llama_70b, "int4")
    assert 36 < w < 39


def test_kv_cache_grows_with_context(llama_70b):
    small = kv_cache_gb(llama_70b, batch=1, seq_len=4096, kv_quant="fp16")
    big = kv_cache_gb(llama_70b, batch=1, seq_len=131072, kv_quant="fp16")
    assert big > small * 30


def test_kv_cache_uses_kv_heads_not_attention_heads(llama_70b):
    # Llama 3.3 70B는 GQA: 64 attention / 8 kv heads
    # 1 × 8192 × 80 × 2 × 8 × 128 × 2 / 1e9 ≈ 2.68 GB
    # 만약 64 heads로 잘못 계산하면 8x 부풀려서 ~21 GB
    kv = kv_cache_gb(llama_70b, batch=1, seq_len=8192, kv_quant="fp16")
    assert 2.0 < kv < 4.0  # GQA 적용된 값
    assert kv < 10.0  # 64 heads로 잘못 계산하면 21 GB가 됨


def test_h100_fits_llama_70b_int4(llama_70b, h100):
    fit = compute_memory_fit(llama_70b, [h100], gpu_count=1,
                             quantization="int4", framework="vllm",
                             context_target=8192, concurrency=1)
    assert fit.fits is True


def test_h100_doesnt_fit_llama_70b_bf16(llama_70b, h100):
    fit = compute_memory_fit(llama_70b, [h100], gpu_count=1,
                             quantization="bf16", framework="vllm",
                             context_target=8192, concurrency=1)
    assert fit.fits is False
    assert fit.fit_status == "no"


def test_4x_rtx4090_total_vram(llama_70b, rtx4090):
    fit = compute_memory_fit(llama_70b, [rtx4090], gpu_count=4,
                             quantization="int4", framework="vllm",
                             context_target=8192, concurrency=1)
    assert fit.total_available_gb == pytest.approx(96, rel=0.01)


def test_rtx4090_single_doesnt_fit_70b_int4(llama_70b, rtx4090):
    fit = compute_memory_fit(llama_70b, [rtx4090], gpu_count=1,
                             quantization="int4", framework="vllm",
                             context_target=8192, concurrency=1)
    assert fit.fits is False
