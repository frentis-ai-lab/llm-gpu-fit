import pytest

from llm_gpu_fit.core.data import load_gpus, load_models
from llm_gpu_fit.core.recommender import (
    is_capability_satisfied,
    pick_quantization,
    recommend,
    suggest_smaller_alternative,
)
from llm_gpu_fit.core.types import UserInput
from llm_gpu_fit.core.usecases import load_use_cases


@pytest.fixture
def gpus():
    return {g.id: g for g in load_gpus()}


@pytest.fixture
def use_cases():
    return {u.id: u for u in load_use_cases()}


@pytest.fixture
def models_by_id():
    return {m.id: m for m in load_models()}


def test_capability_satisfied_with_modalities():
    assert is_capability_satisfied(["text", "tool_use"], ["tool_use"], ["text"])


def test_capability_unsatisfied_when_missing():
    assert not is_capability_satisfied(["text", "tool_use"], ["text"], [])


def test_capability_satisfied_via_modalities_only():
    assert is_capability_satisfied(["text"], ["tool_use"], ["text"])


def test_pick_quantization_for_qwen32_h100(gpus, models_by_id):
    quant = pick_quantization(models_by_id["qwen3_32b"], [gpus["h100_80"]],
                              gpu_count=1, framework="vllm",
                              context_target=8192, concurrency=1)
    assert quant in ("bf16", "fp16")


def test_pick_quantization_falls_back_for_70b_on_h100(gpus, models_by_id):
    # 70B BF16은 안 들어가고, INT8/AWQ/GPTQ/INT4 중 가장 좋은 게 선택
    quant = pick_quantization(models_by_id["llama_3_3_70b"], [gpus["h100_80"]],
                              gpu_count=1, framework="vllm",
                              context_target=8192, concurrency=1)
    assert quant in ("int8", "awq", "gptq", "int4")


def test_recommend_for_coding_h100_returns_results(gpus, use_cases):
    ui = UserInput(use_case="coding", gpu_id="h100_80", gpu_count=1,
                   commercial_required=True, korean_priority=False,
                   onprem_required=False, tool_calling_required=True,
                   concurrency=4, context_target=32768)
    recs = recommend(ui, gpus, use_cases, top_k=3)
    assert len(recs) >= 1
    top_ids = [r.model.id for r in recs]
    assert any(mid in top_ids for mid in ["qwen3_coder_32b", "qwen3_32b"])


def test_recommend_excludes_non_commercial(gpus, use_cases):
    ui = UserInput(use_case="korean_general", gpu_id="rtx4090", gpu_count=1,
                   commercial_required=True, korean_priority=True,
                   onprem_required=False, tool_calling_required=False)
    recs = recommend(ui, gpus, use_cases, top_k=10)
    for r in recs:
        assert r.model.license_commercial_ok is True


def test_recommend_excludes_huge_models_on_4090(gpus, use_cases):
    ui = UserInput(use_case="general", gpu_id="rtx4090", gpu_count=1,
                   commercial_required=False, korean_priority=False,
                   onprem_required=False, tool_calling_required=False)
    recs = recommend(ui, gpus, use_cases, top_k=10)
    ids = [r.model.id for r in recs]
    assert "deepseek_v3_1" not in ids


def test_recommend_excludes_tp_incompatible(gpus, use_cases):
    ui = UserInput(use_case="general", gpu_id="rtx4090", gpu_count=3,
                   commercial_required=False, korean_priority=False,
                   onprem_required=False, tool_calling_required=False)
    recs = recommend(ui, gpus, use_cases, top_k=10)
    for r in recs:
        assert r.topology.tp_compatible is True


def test_alternative_for_multi_gpu(gpus, use_cases):
    ui = UserInput(use_case="coding", gpu_id="rtx4090", gpu_count=4,
                   commercial_required=True, korean_priority=False,
                   onprem_required=False, tool_calling_required=True)
    recs = recommend(ui, gpus, use_cases, top_k=3)
    if not recs:
        pytest.skip("no recs to compare")
    alt = suggest_smaller_alternative(recs[0], ui, gpus, use_cases)
    if alt is not None:
        assert alt.topology.recommended_tp == 1


def test_no_alternative_for_single_gpu(gpus, use_cases):
    ui = UserInput(use_case="coding", gpu_id="h100_80", gpu_count=1,
                   commercial_required=True, korean_priority=False,
                   onprem_required=False, tool_calling_required=True)
    recs = recommend(ui, gpus, use_cases, top_k=3)
    if not recs:
        pytest.skip("no recs to compare")
    alt = suggest_smaller_alternative(recs[0], ui, gpus, use_cases)
    assert alt is None
