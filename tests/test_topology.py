import pytest

from llm_gpu_fit.core.topology import check_topology, recommended_tp_size
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
    return GPU(id="h100_80", name="H100 80GB", vendor="nvidia", vram_gb=80,
               mem_bandwidth_gbs=3350, fp16_tflops=989, int8_tops=1979,
               nvlink=True, form_factor="datacenter")


@pytest.fixture
def rtx4090():
    return GPU(id="rtx4090", name="RTX 4090", vendor="nvidia", vram_gb=24,
               mem_bandwidth_gbs=1008, fp16_tflops=82.6, int8_tops=660,
               nvlink=False, form_factor="consumer")


def test_tp_compatible_when_heads_divisible(llama_70b, h100):
    for n in [1, 2, 4, 8]:
        check = check_topology(llama_70b, h100, gpu_count=n)
        assert check.tp_compatible, f"TP={n} should be compatible"


def test_tp_incompatible_when_heads_not_divisible(llama_70b, rtx4090):
    check = check_topology(llama_70b, rtx4090, gpu_count=3)
    assert check.tp_compatible is False


def test_consumer_multi_gpu_warns_no_nvlink(llama_70b, rtx4090):
    check = check_topology(llama_70b, rtx4090, gpu_count=4)
    assert check.tp_compatible is True
    assert check.nvlink_penalty > 0


def test_datacenter_multi_gpu_no_nvlink_penalty(llama_70b, h100):
    check = check_topology(llama_70b, h100, gpu_count=4)
    assert check.nvlink_penalty == 0


def test_single_gpu_no_warnings(llama_70b, h100):
    check = check_topology(llama_70b, h100, gpu_count=1)
    assert check.recommended_tp == 1
    assert check.nvlink_penalty == 0


def test_recommended_tp_for_small_model_on_big_gpu():
    rec = recommended_tp_size(weights_gb=37, single_gpu_vram=80,
                              available_gpu_count=8, has_nvlink=True)
    assert rec == 1


def test_recommended_tp_for_large_model():
    # 148GB weights + framework + KV → 80GB×2(160GB)는 빠듯, 안전하게 TP=4 권장
    rec = recommended_tp_size(weights_gb=148, single_gpu_vram=80,
                              available_gpu_count=4, has_nvlink=True)
    assert rec in (2, 4)
