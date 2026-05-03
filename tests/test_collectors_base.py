from unittest.mock import MagicMock, patch

from llm_gpu_fit.collectors.base import BaseCollector, CollectorResult
from llm_gpu_fit.collectors.huggingface_hub import HuggingFaceHubCollector


class _FakeCollector(BaseCollector):
    name = "fake"

    def collect(self) -> CollectorResult:
        return CollectorResult(
            models=[],
            benchmarks=[
                {"model_id": "x", "benchmark_id": "y", "score": 50.0,
                 "max_score": 100.0, "normalized_0_100": 50.0,
                 "measured_by": "fake", "confidence": "measured",
                 "measured_at": "2026-05-04", "source_url": "http://e"},
            ],
        )


def test_fake_collector_returns_one_benchmark():
    res = _FakeCollector().collect()
    assert len(res.benchmarks) == 1
    assert res.benchmarks[0]["benchmark_id"] == "y"


def test_hf_hub_extracts_metadata():
    fake_info = MagicMock()
    fake_info.config = {
        "num_attention_heads": 64, "num_key_value_heads": 8,
        "num_hidden_layers": 80, "hidden_size": 8192,
        "max_position_embeddings": 131072,
    }
    fake_info.cardData = {"license": "apache-2.0"}
    fake_info.tags = ["text-generation"]

    with patch("llm_gpu_fit.collectors.huggingface_hub.HfApi") as mock_api:
        mock_api.return_value.model_info.return_value = fake_info
        result = HuggingFaceHubCollector(model_ids=["meta-llama/Llama-3.3-70B-Instruct"]).collect()
        assert len(result.models) == 1
        assert result.models[0]["num_kv_heads"] == 8
        assert result.models[0]["context_window"] == 131072


def test_hf_hub_skips_when_config_missing():
    fake_info = MagicMock()
    fake_info.config = {}
    fake_info.cardData = {}
    fake_info.tags = []

    with patch("llm_gpu_fit.collectors.huggingface_hub.HfApi") as mock_api:
        mock_api.return_value.model_info.return_value = fake_info
        result = HuggingFaceHubCollector(model_ids=["foo/bar"]).collect()
        assert len(result.models) == 0
