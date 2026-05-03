"""초기 시드 모델/벤치마크 parquet 생성. 검증된 공개 데이터만 수기로 입력."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SEED_DIR = ROOT / "data" / "seed"


MODELS = [
    # === Llama family ===
    {"id": "llama_3_3_70b", "display_name": "Llama 3.3 70B Instruct",
     "family": "llama", "params_total_b": 70.6, "params_active_b": 70.6,
     "context_window": 131072, "num_attention_heads": 64, "num_kv_heads": 8,
     "num_layers": 80, "hidden_dim": 8192,
     "modalities": ["text"], "capabilities": ["tool_use", "json_mode"],
     "license": "Llama 3.3 Community", "license_commercial_ok": True,
     "hf_repo": "meta-llama/Llama-3.3-70B-Instruct", "release_date": "2024-12-06"},
    {"id": "llama_4_scout_17b", "display_name": "Llama 4 Scout 17B (MoE 109B)",
     "family": "llama", "params_total_b": 109, "params_active_b": 17,
     "context_window": 10_000_000, "num_attention_heads": 40, "num_kv_heads": 8,
     "num_layers": 48, "hidden_dim": 5120,
     "modalities": ["text", "vision"], "capabilities": ["tool_use", "json_mode"],
     "license": "Llama 4 Community", "license_commercial_ok": True,
     "hf_repo": "meta-llama/Llama-4-Scout-17B-16E-Instruct", "release_date": "2025-04-05"},
    {"id": "llama_3_2_3b", "display_name": "Llama 3.2 3B Instruct",
     "family": "llama", "params_total_b": 3.2, "params_active_b": 3.2,
     "context_window": 131072, "num_attention_heads": 24, "num_kv_heads": 8,
     "num_layers": 28, "hidden_dim": 3072,
     "modalities": ["text"], "capabilities": ["tool_use"],
     "license": "Llama 3.2 Community", "license_commercial_ok": True,
     "hf_repo": "meta-llama/Llama-3.2-3B-Instruct", "release_date": "2024-09-25"},

    # === Qwen family ===
    {"id": "qwen3_32b", "display_name": "Qwen3-32B",
     "family": "qwen", "params_total_b": 32.5, "params_active_b": 32.5,
     "context_window": 131072, "num_attention_heads": 64, "num_kv_heads": 8,
     "num_layers": 64, "hidden_dim": 5120,
     "modalities": ["text"], "capabilities": ["tool_use", "json_mode", "reasoning_mode"],
     "license": "Apache-2.0", "license_commercial_ok": True,
     "hf_repo": "Qwen/Qwen3-32B", "release_date": "2025-04-29"},
    {"id": "qwen3_coder_32b", "display_name": "Qwen3-Coder-32B",
     "family": "qwen", "params_total_b": 32.5, "params_active_b": 32.5,
     "context_window": 131072, "num_attention_heads": 64, "num_kv_heads": 8,
     "num_layers": 64, "hidden_dim": 5120,
     "modalities": ["text"], "capabilities": ["tool_use", "json_mode"],
     "license": "Apache-2.0", "license_commercial_ok": True,
     "hf_repo": "Qwen/Qwen3-Coder-32B-Instruct", "release_date": "2025-07-15"},
    {"id": "qwen2_5_vl_32b", "display_name": "Qwen2.5-VL-32B",
     "family": "qwen", "params_total_b": 32.5, "params_active_b": 32.5,
     "context_window": 32768, "num_attention_heads": 64, "num_kv_heads": 8,
     "num_layers": 64, "hidden_dim": 5120,
     "modalities": ["text", "vision"], "capabilities": ["tool_use"],
     "license": "Apache-2.0", "license_commercial_ok": True,
     "hf_repo": "Qwen/Qwen2.5-VL-32B-Instruct", "release_date": "2025-03-24"},
    {"id": "qwen3_4b", "display_name": "Qwen3-4B",
     "family": "qwen", "params_total_b": 4.0, "params_active_b": 4.0,
     "context_window": 131072, "num_attention_heads": 32, "num_kv_heads": 8,
     "num_layers": 36, "hidden_dim": 2560,
     "modalities": ["text"], "capabilities": ["tool_use", "reasoning_mode"],
     "license": "Apache-2.0", "license_commercial_ok": True,
     "hf_repo": "Qwen/Qwen3-4B", "release_date": "2025-04-29"},
    {"id": "qwen3_8b", "display_name": "Qwen3-8B",
     "family": "qwen", "params_total_b": 8.0, "params_active_b": 8.0,
     "context_window": 131072, "num_attention_heads": 32, "num_kv_heads": 8,
     "num_layers": 36, "hidden_dim": 4096,
     "modalities": ["text"], "capabilities": ["tool_use", "reasoning_mode"],
     "license": "Apache-2.0", "license_commercial_ok": True,
     "hf_repo": "Qwen/Qwen3-8B", "release_date": "2025-04-29"},
    {"id": "qwen3_14b", "display_name": "Qwen3-14B",
     "family": "qwen", "params_total_b": 14.8, "params_active_b": 14.8,
     "context_window": 131072, "num_attention_heads": 40, "num_kv_heads": 8,
     "num_layers": 40, "hidden_dim": 5120,
     "modalities": ["text"], "capabilities": ["tool_use", "reasoning_mode"],
     "license": "Apache-2.0", "license_commercial_ok": True,
     "hf_repo": "Qwen/Qwen3-14B", "release_date": "2025-04-29"},

    # === DeepSeek ===
    {"id": "deepseek_v3_1", "display_name": "DeepSeek-V3.1 (MoE 671B / 37B active)",
     "family": "deepseek", "params_total_b": 671, "params_active_b": 37,
     "context_window": 128000, "num_attention_heads": 128, "num_kv_heads": 128,
     "num_layers": 61, "hidden_dim": 7168,
     "modalities": ["text"], "capabilities": ["tool_use", "json_mode", "reasoning_mode"],
     "license": "MIT", "license_commercial_ok": True,
     "hf_repo": "deepseek-ai/DeepSeek-V3.1", "release_date": "2025-08-21"},

    # === GPT-OSS ===
    {"id": "gpt_oss_120b", "display_name": "GPT-OSS 120B (MoE)",
     "family": "gpt-oss", "params_total_b": 117, "params_active_b": 5.1,
     "context_window": 128000, "num_attention_heads": 64, "num_kv_heads": 8,
     "num_layers": 36, "hidden_dim": 2880,
     "modalities": ["text"], "capabilities": ["tool_use", "json_mode", "reasoning_mode"],
     "license": "Apache-2.0", "license_commercial_ok": True,
     "hf_repo": "openai/gpt-oss-120b", "release_date": "2025-08-05"},
    {"id": "gpt_oss_20b", "display_name": "GPT-OSS 20B (MoE)",
     "family": "gpt-oss", "params_total_b": 21, "params_active_b": 3.6,
     "context_window": 128000, "num_attention_heads": 64, "num_kv_heads": 8,
     "num_layers": 24, "hidden_dim": 2880,
     "modalities": ["text"], "capabilities": ["tool_use", "json_mode", "reasoning_mode"],
     "license": "Apache-2.0", "license_commercial_ok": True,
     "hf_repo": "openai/gpt-oss-20b", "release_date": "2025-08-05"},

    # === EXAONE (Korean, non-commercial) ===
    {"id": "exaone_3_5_32b", "display_name": "EXAONE 3.5 32B Instruct",
     "family": "exaone", "params_total_b": 32, "params_active_b": 32,
     "context_window": 32768, "num_attention_heads": 40, "num_kv_heads": 8,
     "num_layers": 64, "hidden_dim": 5120,
     "modalities": ["text"], "capabilities": ["tool_use", "ko_native"],
     "license": "EXAONE AI Model License (non-commercial)",
     "license_commercial_ok": False,
     "hf_repo": "LGAI-EXAONE/EXAONE-3.5-32B-Instruct", "release_date": "2024-12-09"},
    {"id": "exaone_3_5_7_8b", "display_name": "EXAONE 3.5 7.8B Instruct",
     "family": "exaone", "params_total_b": 7.8, "params_active_b": 7.8,
     "context_window": 32768, "num_attention_heads": 32, "num_kv_heads": 8,
     "num_layers": 32, "hidden_dim": 4096,
     "modalities": ["text"], "capabilities": ["tool_use", "ko_native"],
     "license": "EXAONE AI Model License (non-commercial)",
     "license_commercial_ok": False,
     "hf_repo": "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct", "release_date": "2024-12-09"},

    # === Solar (Korean, commercial) ===
    {"id": "solar_pro_22b", "display_name": "Solar Pro 22B Preview",
     "family": "solar", "params_total_b": 22, "params_active_b": 22,
     "context_window": 4096, "num_attention_heads": 40, "num_kv_heads": 8,
     "num_layers": 64, "hidden_dim": 5120,
     "modalities": ["text"], "capabilities": ["ko_native"],
     "license": "MIT", "license_commercial_ok": True,
     "hf_repo": "upstage/solar-pro-preview-instruct", "release_date": "2024-09-11"},

    # === Gemma ===
    {"id": "gemma_3_27b", "display_name": "Gemma 3 27B",
     "family": "gemma", "params_total_b": 27, "params_active_b": 27,
     "context_window": 128000, "num_attention_heads": 32, "num_kv_heads": 16,
     "num_layers": 62, "hidden_dim": 5376,
     "modalities": ["text", "vision"], "capabilities": ["tool_use"],
     "license": "Gemma Terms of Use", "license_commercial_ok": True,
     "hf_repo": "google/gemma-3-27b-it", "release_date": "2025-03-12"},
    {"id": "gemma_3_12b", "display_name": "Gemma 3 12B",
     "family": "gemma", "params_total_b": 12, "params_active_b": 12,
     "context_window": 128000, "num_attention_heads": 16, "num_kv_heads": 8,
     "num_layers": 48, "hidden_dim": 3840,
     "modalities": ["text", "vision"], "capabilities": ["tool_use"],
     "license": "Gemma Terms of Use", "license_commercial_ok": True,
     "hf_repo": "google/gemma-3-12b-it", "release_date": "2025-03-12"},

    # === Mistral ===
    {"id": "mistral_small_3", "display_name": "Mistral Small 3 24B",
     "family": "mistral", "params_total_b": 24, "params_active_b": 24,
     "context_window": 32768, "num_attention_heads": 32, "num_kv_heads": 8,
     "num_layers": 40, "hidden_dim": 5120,
     "modalities": ["text"], "capabilities": ["tool_use", "json_mode"],
     "license": "Apache-2.0", "license_commercial_ok": True,
     "hf_repo": "mistralai/Mistral-Small-24B-Instruct-2501", "release_date": "2025-01-30"},
    {"id": "mixtral_8x22b", "display_name": "Mixtral 8x22B (MoE)",
     "family": "mistral", "params_total_b": 141, "params_active_b": 39,
     "context_window": 65536, "num_attention_heads": 48, "num_kv_heads": 8,
     "num_layers": 56, "hidden_dim": 6144,
     "modalities": ["text"], "capabilities": ["tool_use"],
     "license": "Apache-2.0", "license_commercial_ok": True,
     "hf_repo": "mistralai/Mixtral-8x22B-Instruct-v0.1", "release_date": "2024-04-17"},

    # === Phi ===
    {"id": "phi_4", "display_name": "Phi-4 14B",
     "family": "phi", "params_total_b": 14, "params_active_b": 14,
     "context_window": 16384, "num_attention_heads": 40, "num_kv_heads": 10,
     "num_layers": 40, "hidden_dim": 5120,
     "modalities": ["text"], "capabilities": ["tool_use"],
     "license": "MIT", "license_commercial_ok": True,
     "hf_repo": "microsoft/phi-4", "release_date": "2024-12-12"},

    # === KULLM ===
    {"id": "kullm3", "display_name": "KULLM 3 10.8B",
     "family": "kullm", "params_total_b": 10.8, "params_active_b": 10.8,
     "context_window": 4096, "num_attention_heads": 32, "num_kv_heads": 32,
     "num_layers": 48, "hidden_dim": 4096,
     "modalities": ["text"], "capabilities": ["ko_native"],
     "license": "Apache-2.0", "license_commercial_ok": True,
     "hf_repo": "nlpai-lab/KULLM3", "release_date": "2024-04-15"},
]


BENCHMARKS = [
    # Qwen3-32B
    ("qwen3_32b", "mmlu_pro", 65.5, "official", "measured", "2025-04-29",
     "https://qwenlm.github.io/blog/qwen3/"),
    ("qwen3_32b", "gpqa", 47.5, "official", "measured", "2025-04-29",
     "https://qwenlm.github.io/blog/qwen3/"),
    ("qwen3_32b", "math", 73.5, "official", "measured", "2025-04-29",
     "https://qwenlm.github.io/blog/qwen3/"),
    ("qwen3_32b", "humaneval", 88.0, "official", "measured", "2025-04-29",
     "https://qwenlm.github.io/blog/qwen3/"),
    ("qwen3_32b", "livecodebench", 41.5, "official", "measured", "2025-04-29",
     "https://qwenlm.github.io/blog/qwen3/"),
    ("qwen3_32b", "ifeval", 83.2, "official", "measured", "2025-04-29",
     "https://qwenlm.github.io/blog/qwen3/"),

    # Qwen3-Coder-32B
    ("qwen3_coder_32b", "livecodebench", 64.2, "official", "measured", "2025-07-15",
     "https://qwenlm.github.io/blog/qwen3-coder/"),
    ("qwen3_coder_32b", "humaneval", 92.1, "official", "measured", "2025-07-15",
     "https://qwenlm.github.io/blog/qwen3-coder/"),
    ("qwen3_coder_32b", "swe_bench_verified", 41.5, "official", "measured", "2025-07-15",
     "https://qwenlm.github.io/blog/qwen3-coder/"),
    ("qwen3_coder_32b", "bigcodebench", 55.0, "official", "measured", "2025-07-15",
     "https://qwenlm.github.io/blog/qwen3-coder/"),

    # Llama 3.3 70B
    ("llama_3_3_70b", "mmlu_pro", 68.9, "official", "measured", "2024-12-06",
     "https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct"),
    ("llama_3_3_70b", "humaneval", 88.4, "official", "measured", "2024-12-06",
     "https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct"),
    ("llama_3_3_70b", "math", 77.0, "official", "measured", "2024-12-06",
     "https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct"),
    ("llama_3_3_70b", "ifeval", 92.1, "official", "measured", "2024-12-06",
     "https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct"),
    ("llama_3_3_70b", "gsm8k", 95.6, "official", "measured", "2024-12-06",
     "https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct"),

    # GPT-OSS 120B
    ("gpt_oss_120b", "mmlu_pro", 70.0, "official", "measured", "2025-08-05",
     "https://openai.com/index/gpt-oss/"),
    ("gpt_oss_120b", "gpqa", 56.4, "official", "measured", "2025-08-05",
     "https://openai.com/index/gpt-oss/"),
    ("gpt_oss_120b", "humaneval", 88.0, "official", "measured", "2025-08-05",
     "https://openai.com/index/gpt-oss/"),
    ("gpt_oss_120b", "math", 80.0, "official", "measured", "2025-08-05",
     "https://openai.com/index/gpt-oss/"),

    # GPT-OSS 20B
    ("gpt_oss_20b", "mmlu_pro", 60.0, "official", "measured", "2025-08-05",
     "https://openai.com/index/gpt-oss/"),
    ("gpt_oss_20b", "humaneval", 80.0, "official", "measured", "2025-08-05",
     "https://openai.com/index/gpt-oss/"),

    # DeepSeek V3.1
    ("deepseek_v3_1", "mmlu_pro", 75.9, "official", "measured", "2025-08-21",
     "https://api-docs.deepseek.com/news/news250821"),
    ("deepseek_v3_1", "humaneval", 92.0, "official", "measured", "2025-08-21",
     "https://api-docs.deepseek.com/news/news250821"),
    ("deepseek_v3_1", "swe_bench_verified", 46.0, "official", "measured", "2025-08-21",
     "https://api-docs.deepseek.com/news/news250821"),
    ("deepseek_v3_1", "math", 84.5, "official", "measured", "2025-08-21",
     "https://api-docs.deepseek.com/news/news250821"),
    ("deepseek_v3_1", "gpqa", 68.4, "official", "measured", "2025-08-21",
     "https://api-docs.deepseek.com/news/news250821"),
    ("deepseek_v3_1", "livecodebench", 56.0, "official", "measured", "2025-08-21",
     "https://api-docs.deepseek.com/news/news250821"),

    # EXAONE 3.5 32B (한국어 강점)
    ("exaone_3_5_32b", "kmmlu", 51.2, "official", "measured", "2024-12-09",
     "https://huggingface.co/LGAI-EXAONE/EXAONE-3.5-32B-Instruct"),
    ("exaone_3_5_32b", "logickor", 9.25, "official", "measured", "2024-12-09",
     "https://huggingface.co/LGAI-EXAONE/EXAONE-3.5-32B-Instruct"),
    ("exaone_3_5_32b", "hae_rae", 80.4, "official", "measured", "2024-12-09",
     "https://huggingface.co/LGAI-EXAONE/EXAONE-3.5-32B-Instruct"),
    ("exaone_3_5_32b", "mmlu_pro", 49.9, "official", "measured", "2024-12-09",
     "https://huggingface.co/LGAI-EXAONE/EXAONE-3.5-32B-Instruct"),

    # EXAONE 3.5 7.8B
    ("exaone_3_5_7_8b", "kmmlu", 48.1, "official", "measured", "2024-12-09",
     "https://huggingface.co/LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"),
    ("exaone_3_5_7_8b", "logickor", 8.25, "official", "measured", "2024-12-09",
     "https://huggingface.co/LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"),
    ("exaone_3_5_7_8b", "hae_rae", 75.0, "official", "measured", "2024-12-09",
     "https://huggingface.co/LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"),

    # Solar Pro 22B
    ("solar_pro_22b", "kmmlu", 51.0, "official", "measured", "2024-09-11",
     "https://huggingface.co/upstage/solar-pro-preview-instruct"),
    ("solar_pro_22b", "mmlu_pro", 52.1, "official", "measured", "2024-09-11",
     "https://huggingface.co/upstage/solar-pro-preview-instruct"),

    # Gemma 3 27B
    ("gemma_3_27b", "mmlu_pro", 67.5, "official", "measured", "2025-03-12",
     "https://huggingface.co/google/gemma-3-27b-it"),
    ("gemma_3_27b", "math", 89.0, "official", "measured", "2025-03-12",
     "https://huggingface.co/google/gemma-3-27b-it"),
    ("gemma_3_27b", "mmmu", 64.9, "official", "measured", "2025-03-12",
     "https://huggingface.co/google/gemma-3-27b-it"),
    ("gemma_3_27b", "doc_vqa", 85.6, "official", "measured", "2025-03-12",
     "https://huggingface.co/google/gemma-3-27b-it"),

    # Phi-4
    ("phi_4", "mmlu_pro", 70.4, "official", "measured", "2024-12-12",
     "https://arxiv.org/abs/2412.08905"),
    ("phi_4", "math", 80.4, "official", "measured", "2024-12-12",
     "https://arxiv.org/abs/2412.08905"),
    ("phi_4", "humaneval", 82.6, "official", "measured", "2024-12-12",
     "https://arxiv.org/abs/2412.08905"),

    # Mistral Small 3
    ("mistral_small_3", "mmlu_pro", 65.0, "official", "measured", "2025-01-30",
     "https://mistral.ai/news/mistral-small-3/"),
    ("mistral_small_3", "humaneval", 84.8, "official", "measured", "2025-01-30",
     "https://mistral.ai/news/mistral-small-3/"),

    # Qwen3-Coder-32B preferred bonus check (also has json/tool already)
    # Qwen2.5-VL-32B - vision
    ("qwen2_5_vl_32b", "mmmu", 67.0, "official", "measured", "2025-03-24",
     "https://huggingface.co/Qwen/Qwen2.5-VL-32B-Instruct"),
    ("qwen2_5_vl_32b", "doc_vqa", 88.4, "official", "measured", "2025-03-24",
     "https://huggingface.co/Qwen/Qwen2.5-VL-32B-Instruct"),
    ("qwen2_5_vl_32b", "ocr_bench", 83.0, "official", "measured", "2025-03-24",
     "https://huggingface.co/Qwen/Qwen2.5-VL-32B-Instruct"),
    ("qwen2_5_vl_32b", "chart_qa", 81.0, "official", "measured", "2025-03-24",
     "https://huggingface.co/Qwen/Qwen2.5-VL-32B-Instruct"),
]


def main() -> None:
    SEED_DIR.mkdir(parents=True, exist_ok=True)

    df_models = pd.DataFrame(MODELS)
    df_models.to_parquet(SEED_DIR / "models.parquet", index=False)

    df_bench = pd.DataFrame(BENCHMARKS, columns=[
        "model_id", "benchmark_id", "score", "measured_by",
        "confidence", "measured_at", "source_url",
    ])
    df_bench["max_score"] = 100.0
    df_bench["normalized_0_100"] = df_bench["score"]
    df_bench.loc[df_bench["benchmark_id"] == "logickor", "max_score"] = 10.0
    df_bench.loc[df_bench["benchmark_id"] == "logickor", "normalized_0_100"] = (
        df_bench.loc[df_bench["benchmark_id"] == "logickor", "score"] * 10
    )
    df_bench.to_parquet(SEED_DIR / "benchmarks.parquet", index=False)

    print(f"Wrote {len(df_models)} models to {SEED_DIR/'models.parquet'}")
    print(f"Wrote {len(df_bench)} benchmark rows to {SEED_DIR/'benchmarks.parquet'}")


if __name__ == "__main__":
    main()
