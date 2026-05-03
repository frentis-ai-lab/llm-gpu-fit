from llm_gpu_fit.core.types import GPU, Model, UserInput


def test_gpu_total_vram_for_multi():
    gpu = GPU(id="rtx4090", name="RTX 4090", vendor="nvidia",
              vram_gb=24, mem_bandwidth_gbs=1008, fp16_tflops=82.6,
              int8_tops=660, nvlink=False, form_factor="consumer")
    assert gpu.total_vram(count=4) == 96


def test_model_is_moe():
    moe = Model(id="dsv3", display_name="DeepSeek V3", family="deepseek",
                params_total_b=671, params_active_b=37, context_window=128_000,
                num_attention_heads=128, num_kv_heads=128, num_layers=61,
                hidden_dim=7168, modalities=["text"], capabilities=["tool_use"],
                license="MIT", license_commercial_ok=True, hf_repo="deepseek-ai/DeepSeek-V3")
    assert moe.is_moe() is True


def test_model_is_not_moe_when_active_equals_total():
    dense = Model(id="qwen32", display_name="Qwen3-32B", family="qwen",
                  params_total_b=32.5, params_active_b=32.5, context_window=131072,
                  num_attention_heads=64, num_kv_heads=8, num_layers=64, hidden_dim=5120,
                  modalities=["text"], capabilities=["tool_use"], license="Apache-2.0",
                  license_commercial_ok=True, hf_repo="Qwen/Qwen3-32B")
    assert dense.is_moe() is False


def test_user_input_default_concurrency():
    ui = UserInput(use_case="coding", gpu_id="h100_80", gpu_count=1,
                   commercial_required=False, korean_priority=False,
                   onprem_required=False, tool_calling_required=False)
    assert ui.concurrency == 1
    assert ui.context_target == 8192
