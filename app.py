"""LLM GPU Fit — HuggingFace Space 진입점."""
import os
import tempfile

import gradio as gr
import yaml

from llm_gpu_fit.core.data import DATA_DIR, FRESH_DIR, load_gpus
from llm_gpu_fit.core.recommender import recommend, suggest_smaller_alternative
from llm_gpu_fit.core.types import UserInput
from llm_gpu_fit.core.usecases import load_use_cases
from llm_gpu_fit.ui.catalog import build_catalog_panel, build_catalog_df
from llm_gpu_fit.ui.matrix import build_matrix_panel
from llm_gpu_fit.ui.results import (
    render_alternative_section, render_plain_text,
    render_recommendation_card, render_results,
)
from llm_gpu_fit.ui.wizard import build_wizard


# HF Space 환경에서 dataset pull
if os.getenv("SPACE_ID"):
    try:
        from huggingface_hub import snapshot_download
        FRESH_DIR.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id="frentis/llm-gpu-fit-data", repo_type="dataset",
            local_dir=str(FRESH_DIR),
        )
        print(f"[startup] pulled fresh data to {FRESH_DIR}")
    except Exception as e:
        print(f"[startup] dataset pull skipped: {e}")


_GPUS_BY_ID = {g.id: g for g in load_gpus()}
_USE_CASES_BY_ID = {u.id: u for u in load_use_cases()}


_NO_REC_MD = ("### 😅 추천할 모델이 없습니다\n\n"
              "이 GPU 구성에 들어가는 모델이 없거나 모든 제약을 만족하는 모델이 없습니다. "
              "GPU 수를 늘리거나 제약 체크를 줄여보세요.")


def _on_submit(mode, summary_mode, use_case, gpu_model, gpu_count,
               commercial, korean, onprem, tool,
               context, concurrency, quant_pref, framework):
    # 간단 모드: 프레임워크는 자동(GPU 종류 기반), 양자화는 사용자 선호 유지
    # (양자화 dropdown 기본값 AWQ를 그대로 적용)
    if mode == "simple":
        framework = ""
    ui = UserInput(
        use_case=use_case, gpu_id=gpu_model, gpu_count=int(gpu_count),
        commercial_required=commercial, korean_priority=korean,
        onprem_required=onprem, tool_calling_required=tool,
        concurrency=int(concurrency), context_target=int(context),
        quantization_preference=quant_pref or None,
        framework=framework or None,
    )
    recs = recommend(ui, _GPUS_BY_ID, _USE_CASES_BY_ID, top_k=3)
    gpu = _GPUS_BY_ID.get(gpu_model)
    sm = summary_mode or "engineer"

    # 3개 순위별 마크다운
    cards = ["", "", ""]
    for i, r in enumerate(recs[:3]):
        cards[i] = render_recommendation_card(r, i + 1, gpu=gpu,
                                              gpu_count=int(gpu_count),
                                              summary_mode=sm)

    # 빈 카드 placeholder
    for i in range(len(recs), 3):
        cards[i] = "*추천 후보 부족*" if recs else _NO_REC_MD if i == 0 else ""

    # 1장 대안
    alt_md = ""
    if recs:
        alt = suggest_smaller_alternative(recs[0], ui, _GPUS_BY_ID, _USE_CASES_BY_ID)
        alt_md = render_alternative_section(recs[0], alt)

    # Slack/이메일 평문
    plain = render_plain_text(recs)

    # 마크다운 다운로드
    md_path = ""
    if recs:
        full_md = render_results(recs, gpu=gpu, gpu_count=int(gpu_count),
                                 summary_mode=sm) + alt_md
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, encoding="utf-8")
        tmp.write(full_md)
        tmp.close()
        md_path = tmp.name

    return cards[0], cards[1], cards[2], alt_md, plain, md_path


def _load_presets() -> list[dict]:
    with (DATA_DIR / "presets.yaml").open(encoding="utf-8") as f:
        return yaml.safe_load(f)["presets"]


def _make_catalog_csv() -> str:
    """전체 모델 카탈로그를 CSV로 임시 파일에 저장 후 경로 반환."""
    df = build_catalog_df("all", "전체", "전체")
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, encoding="utf-8")
    df.to_csv(tmp.name, index=False)
    tmp.close()
    return tmp.name


_CSS = """
.rec-card {
  border: 2px solid #e5e7eb;
  border-radius: 12px;
  padding: 16px;
  background: var(--block-background-fill, #fafafa);
  height: 100%;
  min-height: 350px;
}
.rec-gold { border-color: #f5b400; box-shadow: 0 2px 8px rgba(245,180,0,0.15); }
.rec-silver { border-color: #94a3b8; box-shadow: 0 2px 6px rgba(148,163,184,0.12); }
.rec-bronze { border-color: #b87333; box-shadow: 0 2px 6px rgba(184,115,51,0.12); }
.rec-card h3 { margin-top: 0; }
"""


def main() -> None:
    with gr.Blocks(title="LLM GPU Fit") as demo:
        gr.Markdown(
            "# 🧮 LLM GPU Fit\n\n"
            "용도·GPU·제약을 입력하면 **메모리 적합성·토폴로지·품질 벤치마크·비용**까지 검토해 "
            "알맞은 LLM을 추천합니다."
        )

        with gr.Tab("추천 모드"):
            refs = build_wizard()

            gr.Markdown("### 📊 추천 결과")
            with gr.Row(equal_height=False):
                with gr.Column(min_width=300):
                    card1 = gr.Markdown(elem_classes=["rec-card", "rec-gold"])
                with gr.Column(min_width=300):
                    card2 = gr.Markdown(elem_classes=["rec-card", "rec-silver"])
                with gr.Column(min_width=300):
                    card3 = gr.Markdown(elem_classes=["rec-card", "rec-bronze"])

            alt_md_out = gr.Markdown()

            with gr.Accordion("📋 결과 공유 / 다운로드", open=False):
                plain_box = gr.Textbox(
                    label="Slack·이메일에 붙여넣을 평문",
                    buttons=["copy"], lines=10, interactive=False,
                )
                md_file = gr.File(label="마크다운 다운로드 (.md)",
                                  interactive=False)

            refs["submit"].click(
                _on_submit,
                inputs=[refs["mode"], refs["summary_mode"],
                        refs["use_case"], refs["gpu_model"], refs["gpu_count"],
                        refs["commercial"], refs["korean"], refs["onprem"], refs["tool"],
                        refs["context"], refs["concurrency"],
                        refs["quant_pref"], refs["framework"]],
                outputs=[card1, card2, card3, alt_md_out, plain_box, md_file],
            )

            gr.Markdown("---\n### 자주 묻는 시나리오\n버튼 한 번으로 양식이 채워지고 추천이 실행됩니다.")
            for preset in _load_presets():
                with gr.Row():
                    with gr.Column(scale=4):
                        gr.Markdown(f"**{preset['title']}** — {preset['description']}")
                    with gr.Column(scale=1):
                        btn = gr.Button("적용하고 추천", size="sm")

                        def _apply(p_use=preset["use_case"],
                                   p_gpu=preset["gpu_id"],
                                   p_count=preset["gpu_count"],
                                   p_com=preset["commercial_required"],
                                   p_ko=preset["korean_priority"],
                                   p_op=preset["onprem_required"],
                                   p_tl=preset["tool_calling_required"],
                                   p_ctx=preset["context_target"],
                                   p_cc=preset["concurrency"]):
                            c1, c2, c3, alt, plain, md_path = _on_submit(
                                "simple", "engineer",
                                p_use, p_gpu, p_count, p_com, p_ko,
                                p_op, p_tl, p_ctx, p_cc, "awq", "")
                            return ("simple", "engineer",
                                    p_use, p_gpu, p_count, p_com, p_ko, p_op, p_tl,
                                    p_ctx, p_cc, "awq", "",
                                    c1, c2, c3, alt, plain, md_path)
                        btn.click(_apply, inputs=[], outputs=[
                            refs["mode"], refs["summary_mode"],
                            refs["use_case"], refs["gpu_model"], refs["gpu_count"],
                            refs["commercial"], refs["korean"], refs["onprem"], refs["tool"],
                            refs["context"], refs["concurrency"],
                            refs["quant_pref"], refs["framework"],
                            card1, card2, card3, alt_md_out, plain_box, md_file,
                        ])

        with gr.Tab("매트릭스 모드"):
            build_matrix_panel()

        with gr.Tab("모델 카탈로그"):
            build_catalog_panel()
            with gr.Row():
                csv_btn = gr.Button("📥 전체 카탈로그 CSV 다운로드", size="sm")
                csv_file = gr.File(label="", interactive=False)
                csv_btn.click(_make_catalog_csv, inputs=[], outputs=csv_file)

        with gr.Tab("API 사용"):
            gr.Markdown("""
### 🔌 REST API로 자동화

이 Space는 Gradio 자동 생성 API를 제공합니다. Slack bot, 사내 도구에 연동 가능.

**Python 예시**:
```python
from gradio_client import Client

client = Client("frentis/llm-gpu-fit")
result = client.predict(
    "simple",       # mode
    "engineer",     # summary_mode
    "coding",       # use_case
    "h100_80",      # gpu_model
    1,              # gpu_count
    True,           # commercial_required
    False,          # korean_priority
    False,          # onprem_required
    True,           # tool_calling_required
    32768,          # context
    4,              # concurrency
    "",             # quant_pref
    "",             # framework
    api_name="/predict",
)
print(result[1])  # 평문 추천 결과
```

**curl 예시**: API 문서는 우측 상단 `Use via API` 또는
`https://frentis-llm-gpu-fit.hf.space/?view=api`에서 확인.
            """)

        gr.Markdown(
            "---\n"
            "출처: 모델 카드 + Open Ko-LLM Leaderboard + 공식 발표값. "
            "데이터는 매주 자동 갱신. "
            "[GitHub](https://github.com/frentis-ai-lab/llm-gpu-fit) · "
            "[Spec](https://github.com/frentis-ai-lab/llm-gpu-fit/blob/main/docs/spec.md)"
        )

    demo.launch(server_name="0.0.0.0", server_port=7860,
                theme=gr.themes.Soft(), css=_CSS)


if __name__ == "__main__":
    main()
