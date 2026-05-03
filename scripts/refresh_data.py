"""주간 cron 호출 진입점. collectors → parquet 병합 → HF Dataset push."""
import os
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi

from llm_gpu_fit.collectors.huggingface_hub import HuggingFaceHubCollector


ROOT = Path(__file__).resolve().parent.parent
SEED_DIR = ROOT / "data" / "seed"
OUT_DIR = ROOT / "data" / "fresh"
DATASET_REPO = os.getenv("HF_DATASET_REPO", "frentis/llm-gpu-fit-data")


def _watched_models() -> list[str]:
    df = pd.read_parquet(SEED_DIR / "models.parquet")
    return df["hf_repo"].tolist()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    collectors = [
        HuggingFaceHubCollector(model_ids=_watched_models()),
    ]

    all_models: list[dict] = []
    all_benchmarks: list[dict] = []
    for c in collectors:
        print(f"[refresh] running {c.name}")
        res = c.collect()
        all_models.extend(res.models)
        all_benchmarks.extend(res.benchmarks)
        print(f"  → {len(res.models)} models, {len(res.benchmarks)} benchmarks")

    seed_models = pd.read_parquet(SEED_DIR / "models.parquet")
    seed_bench = pd.read_parquet(SEED_DIR / "benchmarks.parquet")

    fresh_models = pd.DataFrame(all_models) if all_models else pd.DataFrame()
    fresh_bench = pd.DataFrame(all_benchmarks) if all_benchmarks else pd.DataFrame()

    if not fresh_models.empty:
        merged_models = pd.concat([fresh_models, seed_models]).drop_duplicates(
            subset=["id"], keep="first")
    else:
        merged_models = seed_models

    if not fresh_bench.empty:
        merged_bench = pd.concat([fresh_bench, seed_bench]).drop_duplicates(
            subset=["model_id", "benchmark_id"], keep="first")
    else:
        merged_bench = seed_bench

    merged_models.to_parquet(OUT_DIR / "models.parquet", index=False)
    merged_bench.to_parquet(OUT_DIR / "benchmarks.parquet", index=False)
    print(f"[refresh] wrote {len(merged_models)} models, {len(merged_bench)} benchmarks")

    if os.getenv("PUSH_TO_HF") == "1":
        api = HfApi(token=os.environ["HF_TOKEN"])
        api.create_repo(DATASET_REPO, repo_type="dataset", exist_ok=True)
        api.upload_folder(
            folder_path=str(OUT_DIR),
            repo_id=DATASET_REPO,
            repo_type="dataset",
            commit_message="data refresh (cron)",
        )
        print(f"[refresh] pushed to {DATASET_REPO}")


if __name__ == "__main__":
    main()
