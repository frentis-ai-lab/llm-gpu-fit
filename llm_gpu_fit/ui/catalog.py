"""모델 카탈로그 — DB에 등록된 모든 모델 정보 + 벤치마크 한눈에 보기."""
from collections import defaultdict
from typing import Literal

import gradio as gr
import pandas as pd

from llm_gpu_fit.core.data import load_benchmarks_for, load_models


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


_LANG_FLAGS = {
    "en": "🇺🇸", "ko": "🇰🇷", "zh": "🇨🇳", "ja": "🇯🇵",
    "es": "🇪🇸", "fr": "🇫🇷", "de": "🇩🇪", "it": "🇮🇹",
    "pt": "🇵🇹", "ru": "🇷🇺", "ar": "🇸🇦", "hi": "🇮🇳",
    "vi": "🇻🇳", "th": "🇹🇭", "id": "🇮🇩", "tr": "🇹🇷",
    "nl": "🇳🇱", "pl": "🇵🇱", "he": "🇮🇱", "fa": "🇮🇷",
    "el": "🇬🇷", "ro": "🇷🇴", "uk": "🇺🇦", "cs": "🇨🇿",
}


def _languages_badge(langs: tuple[str, ...]) -> str:
    if not langs:
        return "🇺🇸"  # 기본 영문
    multi = next((lang for lang in langs if lang.startswith("multi-")), None)
    if multi:
        n = multi.split("-")[1]
        flags = [_LANG_FLAGS[lang] for lang in langs if lang in _LANG_FLAGS][:5]
        return f"🌍×{n} " + "".join(flags)
    flags = [_LANG_FLAGS[lang] for lang in langs if lang in _LANG_FLAGS]
    return "".join(flags) if flags else "🇺🇸"


def _korean_strength_badge(strength: str) -> str:
    return {
        "native": "🇰🇷 네이티브",
        "multilingual": "🇰🇷 다국어",
        "partial": "🇰🇷? 다국어100+",
        "none": "—",
    }.get(strength, "—")


def _commercial_str(model) -> str:
    return "✓상용" if model.license_commercial_ok else "✗비상용"


def _popularity_badge(tier: int) -> str:
    return {5: "🔥 핫", 4: "⭐ 인기", 3: "✓ 일반",
            2: "⏳ 구", 1: "📦 레거시"}.get(tier, "")


FilterMode = Literal["all", "korean", "korean_native", "korean_multilingual",
                     "vision", "audio", "moe", "commercial",
                     "non_commercial", "small", "medium", "large"]


def build_catalog_df(filter_mode: FilterMode = "all",
                     company_filter: str = "전체",
                     series_filter: str = "전체") -> pd.DataFrame:
    rows = []
    for model in load_models():
        if company_filter != "전체" and model.company_or_family != company_filter:
            continue
        if series_filter != "전체" and model.series_or_family != series_filter:
            continue
        if filter_mode == "korean" and not model.supports_korean:
            continue
        if filter_mode == "korean_native" and model.korean_strength != "native":
            continue
        if filter_mode == "korean_multilingual" and model.korean_strength != "multilingual":
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
            "인기": _popularity_badge(model.popularity_tier),
            "시리즈": model.series_or_family,
            "회사": model.company_or_family,
            "출시일": model.release_date or "—",
            "파라미터": _params_str(model.params_total_b, model.params_active_b),
            "컨텍스트": _ctx_str(model.context_window),
            "모달": _modality_badge(model.modalities),
            "언어": _languages_badge(model.languages),
            "한국어": _korean_strength_badge(model.korean_strength),
            "기능": _capability_badge(model.capabilities),
            "라이선스": _commercial_str(model),
            "벤치 수": len(bench),
        }
        for bid, label in _KEY_BENCHES:
            row[label] = round(bench[bid], 1) if bid in bench else None
        row["HF Repo"] = model.hf_repo
        rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        # 인기도 → 출시일 → 파라미터 순 정렬
        df = df.sort_values(["인기", "출시일", "파라미터"],
                            ascending=[False, False, False])
    return df


def build_series_summary_df(filter_mode: FilterMode = "all",
                            company_filter: str = "전체") -> pd.DataFrame:
    by_series: dict[str, list] = defaultdict(list)
    for model in load_models():
        if company_filter != "전체" and model.company_or_family != company_filter:
            continue
        if filter_mode == "korean" and not model.supports_korean:
            continue
        if filter_mode == "korean_native" and model.korean_strength != "native":
            continue
        if filter_mode == "korean_multilingual" and model.korean_strength != "multilingual":
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
        by_series[model.series_or_family].append(model)

    rows = []
    for series, variants in by_series.items():
        sizes = sorted({v.params_total_b for v in variants})
        size_str = ", ".join(f"{s:g}B" for s in sizes)
        # 가장 큰 모델 기준으로 메타 추출
        rep = max(variants, key=lambda v: v.params_total_b)
        ctx_max = max(v.context_window for v in variants)
        modalities_union = set()
        capabilities_union = set()
        for v in variants:
            modalities_union.update(v.modalities)
            capabilities_union.update(v.capabilities)
        latest_date = max((v.release_date for v in variants if v.release_date),
                          default="—")
        bench_total = sum(len(load_benchmarks_for(v.id)) for v in variants)
        max_pop = max(v.popularity_tier for v in variants)
        rows.append({
            "시리즈": series,
            "인기": _popularity_badge(max_pop),
            "회사": rep.company_or_family,
            "변형 수": len(variants),
            "사이즈 라인업": size_str,
            "최대 컨텍스트": _ctx_str(ctx_max),
            "모달": _modality_badge(sorted(modalities_union,
                                          key=lambda x: ["text", "vision", "audio"].index(x)
                                          if x in ["text", "vision", "audio"] else 99)),
            "언어": _languages_badge(rep.languages),
            "한국어": _korean_strength_badge(rep.korean_strength),
            "기능": _capability_badge(sorted(capabilities_union)),
            "라이선스": _commercial_str(rep),
            "최신 출시": latest_date,
            "벤치 합계": bench_total,
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["인기", "최신 출시", "변형 수"],
                            ascending=[False, False, False])
    return df


def company_choices() -> list[str]:
    companies = sorted({m.company_or_family for m in load_models()})
    return ["전체"] + companies


def series_choices() -> list[str]:
    series_set = sorted({m.series_or_family for m in load_models()})
    return ["전체"] + series_set


def build_catalog_panel() -> dict:
    total_models = len(load_models())
    total_benches = sum(len(load_benchmarks_for(m.id)) for m in load_models())
    total_series = len({m.series_or_family for m in load_models()})
    total_companies = len({m.company_or_family for m in load_models()})

    gr.Markdown(
        "### 📚 모델 카탈로그\n"
        f"**{total_models}개 모델 · {total_series}개 시리즈 · "
        f"{total_companies}개 회사 · 벤치마크 {total_benches}행**\n\n"
        "DB에 등록된 모든 모델의 사양·벤치마크·라이선스를 확인합니다. "
        "**시리즈별 보기**로 같은 패밀리 모델을 묶어서 보거나, "
        "**모델별 보기**로 변형 단위 정렬·필터링."
    )

    with gr.Tab("시리즈별 보기"):
        gr.Markdown("같은 시리즈(예: Qwen3, EXAONE Deep, Llama 3.3)를 묶어 표시합니다.")
        with gr.Row():
            s_filter = gr.Dropdown(
                label="속성 필터",
                choices=[
                    ("전체", "all"),
                    ("🇰🇷 한국어 지원", "korean"),
                    ("🇰🇷 한국어 네이티브", "korean_native"),
                    ("🇰🇷 한국어 다국어", "korean_multilingual"),
                    ("👁 비전 지원", "vision"),
                    ("🎙 음성 지원", "audio"),
                    ("MoE 시리즈", "moe"),
                    ("✓ 상용 가능", "commercial"),
                    ("✗ 비상용", "non_commercial"),
                ],
                value="all",
            )
            s_company = gr.Dropdown(label="회사", choices=company_choices(), value="전체")
        s_table = gr.Dataframe(
            value=build_series_summary_df("all", "전체"),
            label=f"{total_series}개 시리즈 · 최신 출시순",
            interactive=False, wrap=True, max_height=600,
        )

        def _s_refresh(f, c):
            return build_series_summary_df(f, c)
        s_filter.change(_s_refresh, inputs=[s_filter, s_company], outputs=s_table)
        s_company.change(_s_refresh, inputs=[s_filter, s_company], outputs=s_table)

    with gr.Tab("모델별 보기 (변형)"):
        gr.Markdown("개별 모델 변형(예: Qwen3-32B, Qwen3-14B)을 행 단위로 표시합니다.")
        with gr.Row():
            filter_mode = gr.Dropdown(
                label="속성 필터",
                choices=[
                    ("전체 모델", "all"),
                    ("🇰🇷 한국어 지원 (네이티브+다국어)", "korean"),
                    ("🇰🇷 한국어 네이티브 전용", "korean_native"),
                    ("🇰🇷 한국어 포함 다국어", "korean_multilingual"),
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
            company = gr.Dropdown(label="회사", choices=company_choices(), value="전체")
            series = gr.Dropdown(label="시리즈", choices=series_choices(), value="전체")
        table = gr.Dataframe(
            value=build_catalog_df("all", "전체", "전체"),
            label=f"{total_models}개 모델 · 시리즈 그룹 정렬",
            interactive=False, wrap=True, max_height=600,
        )

        def _refresh(f, c, s):
            return build_catalog_df(f, c, s)
        filter_mode.change(_refresh, inputs=[filter_mode, company, series], outputs=table)
        company.change(_refresh, inputs=[filter_mode, company, series], outputs=table)
        series.change(_refresh, inputs=[filter_mode, company, series], outputs=table)

    legend = gr.Markdown(
        "**범례**: 📝 텍스트 · 👁 비전 · 🎙 음성 · "
        "🔧 Tool calling · 📋 JSON mode · 🧠 Reasoning mode · 🇰🇷 한국어 네이티브\n\n"
        "**시리즈** = 같은 베이스 모델의 사이즈/변형 (Qwen3는 4B/8B/14B/32B). "
        "벤치마크는 **공식 발표값** 기준. 매주 cron이 collector로 확장합니다.\n\n"
        "기여: [GitHub repo](https://github.com/frentis-ai-lab/llm-gpu-fit)의 "
        "`scripts/seed_data.py`에 PR."
    )

    return {"legend": legend}
