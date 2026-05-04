"""모델 카탈로그 — DB에 등록된 모든 모델 정보 + 벤치마크 한눈에 보기."""
from typing import Literal

import gradio as gr
import pandas as pd

from llm_gpu_fit.core.data import load_benchmarks_for, load_models


# 카탈로그에 보여줄 핵심 벤치마크 (있는 모델만 점수, 없으면 빈칸)
_KEY_BENCHES = [
    ("mmlu_pro", "MMLU-Pro"),
    ("gpqa", "GPQA"),
    ("math", "MATH"),
    ("aime", "AIME"),
    ("humaneval", "HumanEval"),
    ("livecodebench", "LiveCodeBench"),
    ("swe_bench_verified", "SWE-bench"),
    ("bfcl_v3", "BFCL"),
    ("ifeval", "IFEval"),
    ("kmmlu", "KMMLU"),
    ("hae_rae", "HAE-RAE"),
    ("logickor", "LogicKor"),
    ("mmmu", "MMMU"),
    ("ocr_bench", "OCR"),
    ("doc_vqa", "DocVQA"),
    ("ruler", "RULER"),
]


def _params_str(total: float, active: float) -> str:
    if active < total:
        return f"{total:g}B (활성 {active:g}B, MoE)"
    return f"{total:g}B"


def _ctx_str(ctx: int) -> str:
    if ctx >= 1_000_000:
        return f"{ctx/1_000_000:.1f}M"
    if ctx >= 1000:
        return f"{ctx//1000}K"
    return str(ctx)


def _modality_badge(mods: list[str]) -> str:
    icons = {"text": "📝", "vision": "👁", "audio": "🎙"}
    return " ".join(icons.get(m, m) for m in mods)


def _capability_badge(caps: list[str]) -> str:
    icons = {
        "tool_use": "🔧", "json_mode": "📋",
        "reasoning_mode": "🧠", "ko_native": "🇰🇷",
    }
    return " ".join(icons[c] for c in caps if c in icons) or "—"


def _commercial_str(model) -> str:
    return "✓상용" if model.license_commercial_ok else "✗비상용"


FilterMode = Literal["all", "korean", "vision", "audio", "moe", "commercial",
                     "non_commercial", "small", "medium", "large"]


def build_catalog_df(filter_mode: FilterMode = "all",
                     family_filter: str = "전체") -> pd.DataFrame:
    rows = []
    for model in load_models():
        if family_filter != "전체" and model.family != family_filter:
            continue
        if filter_mode == "korean" and "ko_native" not in model.capabilities:
            continue
        if filter_mode == "vision" and "vision" not in model.modalities:
            continue
        if filter_mode == "audio" and "audio" not in model.modalities:
            continue
        if filter_mode == "moe" and not model.is_moe():
            continue
        if filter_mode == "commercial" and not model.license_commercial_ok:
            continue
        if filter_mode == "non_commercial" and model.license_commercial_ok:
            continue
        if filter_mode == "small" and model.params_total_b > 10:
            continue
        if filter_mode == "medium" and (model.params_total_b <= 10
                                        or model.params_total_b > 50):
            continue
        if filter_mode == "large" and model.params_total_b <= 50:
            continue

        bench = load_benchmarks_for(model.id)
        row = {
            "모델": model.display_name,
            "패밀리": model.family,
            "파라미터": _params_str(model.params_total_b, model.params_active_b),
            "컨텍스트": _ctx_str(model.context_window),
            "모달": _modality_badge(model.modalities),
            "기능": _capability_badge(model.capabilities),
            "라이선스": _commercial_str(model),
            "출시": model.release_date,
            "벤치 수": len(bench),
        }
        for bid, label in _KEY_BENCHES:
            row[label] = round(bench[bid], 1) if bid in bench else None
        row["HF Repo"] = model.hf_repo
        rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["벤치 수", "파라미터"], ascending=[False, False])
    return df


def family_choices() -> list[str]:
    families = sorted({m.family for m in load_models()})
    return ["전체"] + families


def build_catalog_panel() -> dict:
    gr.Markdown(
        "### 📚 모델 카탈로그\n"
        f"**총 {len(load_models())}개 모델 · 벤치마크 {sum(len(load_benchmarks_for(m.id)) for m in load_models())}행 등록**\n\n"
        "DB에 등록된 모든 모델의 사양·벤치마크·라이선스를 한눈에 확인합니다. "
        "필터로 좁히거나, 컬럼 클릭으로 정렬, 가로 스크롤로 모든 벤치마크 점수 확인."
    )

    with gr.Row():
        filter_mode = gr.Dropdown(
            label="필터",
            choices=[
                ("전체 모델", "all"),
                ("🇰🇷 한국어 네이티브", "korean"),
                ("👁 비전 지원", "vision"),
                ("🎙 음성 지원", "audio"),
                ("MoE 모델", "moe"),
                ("✓ 상용 가능", "commercial"),
                ("✗ 비상용 (연구용)", "non_commercial"),
                ("소형 (≤10B)", "small"),
                ("중형 (10B-50B)", "medium"),
                ("대형 (>50B)", "large"),
            ],
            value="all",
        )
        family = gr.Dropdown(label="패밀리", choices=family_choices(), value="전체")

    table = gr.Dataframe(
        value=build_catalog_df("all", "전체"),
        label="",
        interactive=False, wrap=True,
        max_height=600,
    )

    legend = gr.Markdown(
        "**범례**: 📝 텍스트 · 👁 비전 · 🎙 음성 · "
        "🔧 Tool calling · 📋 JSON mode · 🧠 Reasoning mode · 🇰🇷 한국어 네이티브\n\n"
        "벤치마크 점수는 **공식 발표값** 기준 (모델 카드/공식 블로그). "
        "빈 칸은 발표값이 없거나 미수집 — 매주 cron이 collector를 돌려 확장합니다.\n\n"
        "기여 환영: [GitHub repo](https://github.com/frentis-ai-lab/llm-gpu-fit)에서 "
        "`scripts/seed_data.py`의 `BENCHMARKS`에 행 추가 PR."
    )

    def _refresh(f_mode, fam):
        return build_catalog_df(f_mode, fam)

    filter_mode.change(_refresh, inputs=[filter_mode, family], outputs=table)
    family.change(_refresh, inputs=[filter_mode, family], outputs=table)

    return {"filter": filter_mode, "family": family,
            "table": table, "legend": legend}
