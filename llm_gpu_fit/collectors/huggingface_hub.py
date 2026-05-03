import os

from huggingface_hub import HfApi

from llm_gpu_fit.collectors.base import BaseCollector, CollectorResult


_REQUIRED_CFG_KEYS = ("num_attention_heads", "num_hidden_layers", "hidden_size")


class HuggingFaceHubCollector(BaseCollector):
    name = "huggingface_hub"

    def __init__(self, model_ids: list[str], token: str | None = None) -> None:
        self.model_ids = model_ids
        self.token = token or os.getenv("HF_TOKEN")

    def collect(self) -> CollectorResult:
        api = HfApi(token=self.token)
        models: list[dict] = []
        for repo_id in self.model_ids:
            try:
                info = api.model_info(repo_id, expand=["config", "cardData", "tags"])
            except Exception as e:
                print(f"[hf_hub] skip {repo_id}: {e}")
                continue
            cfg = info.config or {}
            if not all(k in cfg for k in _REQUIRED_CFG_KEYS):
                continue
            card = info.cardData or {}
            models.append({
                "id": repo_id.replace("/", "__"),
                "hf_repo": repo_id,
                "display_name": repo_id.split("/")[-1],
                "family": repo_id.split("/")[0].lower(),
                "num_attention_heads": int(cfg["num_attention_heads"]),
                "num_kv_heads": int(cfg.get("num_key_value_heads", cfg["num_attention_heads"])),
                "num_layers": int(cfg["num_hidden_layers"]),
                "hidden_dim": int(cfg["hidden_size"]),
                "context_window": int(cfg.get("max_position_embeddings", 4096)),
                "license": card.get("license", "unknown"),
            })
        return CollectorResult(models=models)
