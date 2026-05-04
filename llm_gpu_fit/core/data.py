from pathlib import Path

import pandas as pd
import yaml

from llm_gpu_fit.core.types import GPU, Model


DATA_DIR = Path(__file__).resolve().parents[2] / "data"
SEED_DIR = DATA_DIR / "seed"
FRESH_DIR = DATA_DIR / "fresh"


def _data_dir() -> Path:
    return FRESH_DIR if (FRESH_DIR / "models.parquet").exists() else SEED_DIR


def load_gpus() -> list[GPU]:
    with (DATA_DIR / "gpus.yaml").open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return [GPU(**g) for g in raw["gpus"]]


def _model_from_row(row: dict) -> Model:
    return Model(
        id=row["id"],
        display_name=row["display_name"],
        family=row["family"],
        params_total_b=float(row["params_total_b"]),
        params_active_b=float(row["params_active_b"]),
        context_window=int(row["context_window"]),
        num_attention_heads=int(row["num_attention_heads"]),
        num_kv_heads=int(row["num_kv_heads"]),
        num_layers=int(row["num_layers"]),
        hidden_dim=int(row["hidden_dim"]),
        modalities=list(row["modalities"]),
        capabilities=list(row["capabilities"]),
        license=row["license"],
        license_commercial_ok=bool(row["license_commercial_ok"]),
        hf_repo=row["hf_repo"],
        release_date=row.get("release_date", ""),
        company=row.get("company", "") or "",
        series=row.get("series", "") or "",
        languages=tuple(row["languages"]) if "languages" in row and row["languages"] is not None and len(row["languages"]) > 0 else (),
        popularity_tier=int(row.get("popularity_tier", 3) or 3),
    )


def load_models() -> list[Model]:
    df = pd.read_parquet(_data_dir() / "models.parquet")
    return [_model_from_row(r) for r in df.to_dict("records")]


def load_benchmarks_for(model_id: str) -> dict[str, float]:
    path = _data_dir() / "benchmarks.parquet"
    if not path.exists():
        return {}
    df = pd.read_parquet(path)
    rows = df[df["model_id"] == model_id]
    return {r["benchmark_id"]: float(r["normalized_0_100"]) for _, r in rows.iterrows()}
