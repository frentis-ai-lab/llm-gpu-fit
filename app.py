"""LLM GPU Fit — HuggingFace Space 진입점."""
import os

import gradio as gr
import yaml

from llm_gpu_fit.core.data import DATA_DIR, FRESH_DIR, load_gpus
from llm_gpu_fit.core.recommender import recommend, suggest_smaller_alternative
from llm_gpu_fit.core.types import UserInput
from llm_gpu_fit.core.usecases import load_use_cases
from llm_gpu_fit.ui.matrix import build_matrix_panel
from llm_gpu_fit.ui.results import render_alternative_section, render_results
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


def _on_submit(use_case, gpu_selection, commercial, korean, onprem, tool,
               context, concurrency, quant_pref):
    gpu_id, gpu_count_str = gpu_selection.split("::")
    gpu_count = int(gpu_count_str)
    ui = UserInput(
        use_case=use_case, gpu_id=gpu_id, gpu_count=gpu_count,
        commercial_required=commercial, korean_priority=korean,
        onprem_required=onprem, tool_calling_required=tool,
        concurrency=int(concurrency), context_target=int(context),
        quantization_preference=quant_pref or None,
    )
    recs = recommend(ui, _GPUS_BY_ID, _USE_CASES_BY_ID, top_k=3)
    main_md = render_results(recs)
    if recs:
        alt = suggest_smaller_alternative(recs[0], ui, _GPUS_BY_ID, _USE_CASES_BY_ID)
        return main_md + render_alternative_section(recs[0], alt)
    return main_md


def _load_presets() -> list[dict]:
    with (DATA_DIR / "presets.yaml").open(encoding="utf-8") as f:
        return yaml.safe_load(f)["presets"]


def main() -> None:
    with gr.Blocks(title="LLM GPU Fit") as demo:
        gr.Markdown(
            "# 🧮 LLM GPU Fit\n\n"
            "메모리 적합성·토폴로지·품질 벤치마크를 한 번에 검토해 알맞은 모델을 추천합니다.\n"
            "용도와 GPU 구성, 조직 제약만 입력하세요."
        )
        with gr.Tab("추천 모드"):
            refs = build_wizard()
            results = gr.Markdown()
            refs["submit"].click(
                _on_submit,
                inputs=[refs["use_case"], refs["gpu_selection"],
                        refs["commercial"], refs["korean"], refs["onprem"], refs["tool"],
                        refs["context"], refs["concurrency"], refs["quant_pref"]],
                outputs=results,
            )

            gr.Markdown("---\n### 자주 묻는 시나리오")
            for preset in _load_presets():
                with gr.Row():
                    with gr.Column(scale=4):
                        gr.Markdown(f"**{preset['title']}** — {preset['description']}")
                    with gr.Column(scale=1):
                        btn = gr.Button("적용하고 추천", size="sm")
                        gpu_sel = f"{preset['gpu_id']}::{preset['gpu_count']}"

                        def _apply(p_use=preset["use_case"], p_gpu=gpu_sel,
                                   p_com=preset["commercial_required"],
                                   p_ko=preset["korean_priority"],
                                   p_op=preset["onprem_required"],
                                   p_tl=preset["tool_calling_required"],
                                   p_ctx=preset["context_target"],
                                   p_cc=preset["concurrency"]):
                            return (p_use, p_gpu, p_com, p_ko, p_op, p_tl,
                                    p_ctx, p_cc, "",
                                    _on_submit(p_use, p_gpu, p_com, p_ko, p_op,
                                               p_tl, p_ctx, p_cc, ""))
                        btn.click(_apply, inputs=[], outputs=[
                            refs["use_case"], refs["gpu_selection"],
                            refs["commercial"], refs["korean"], refs["onprem"], refs["tool"],
                            refs["context"], refs["concurrency"], refs["quant_pref"],
                            results,
                        ])

        with gr.Tab("매트릭스 모드"):
            build_matrix_panel()

        gr.Markdown(
            "---\n"
            "출처: 모델 카드 + Open Ko-LLM Leaderboard + 공식 발표값. "
            "데이터는 `data/seed/`에 시드, 매주 자동 갱신. "
            "[GitHub](https://github.com/frentis-ai-lab/llm-gpu-fit) · "
            "[Spec](https://github.com/frentis-ai-lab/llm-gpu-fit/blob/main/docs/spec.md)"
        )

    demo.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft())


if __name__ == "__main__":
    main()
