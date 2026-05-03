import gradio as gr
import pandas as pd

from llm_gpu_fit.core.data import load_benchmarks_for, load_gpus, load_models
from llm_gpu_fit.core.memory import compute_memory_fit
from llm_gpu_fit.core.recommender import is_capability_satisfied, pick_quantization
from llm_gpu_fit.core.topology import check_topology
from llm_gpu_fit.core.usecases import load_use_cases, score_quality


def build_matrix(use_case_id: str, gpu_id: str, gpu_count: int,
                 context_target: int = 8192, concurrency: int = 1) -> pd.DataFrame:
    use_cases = {u.id: u for u in load_use_cases()}
    gpus = {g.id: g for g in load_gpus()}
    if use_case_id not in use_cases or gpu_id not in gpus:
        return pd.DataFrame()
    use_case = use_cases[use_case_id]
    gpu = gpus[gpu_id]
    framework = "mlx" if gpu.form_factor == "apple_silicon" else "vllm"

    rows = []
    for model in load_models():
        if not is_capability_satisfied(use_case.required_capabilities,
                                       model.capabilities, model.modalities):
            continue
        quant = pick_quantization(model, [gpu], gpu_count, framework,
                                  context_target, concurrency)
        if quant is None:
            rows.append({
                "모델": model.display_name, "양자화": "—",
                "메모리": "❌ 안 들어감", "최대 컨텍스트": 0,
                "품질 점수": None,
                "라이선스": "✓상용" if model.license_commercial_ok else "✗비상용",
            })
            continue
        fit = compute_memory_fit(model, [gpu], gpu_count, quant, framework,
                                 context_target, concurrency)
        topo = check_topology(model, gpu, gpu_count)
        bench = load_benchmarks_for(model.id)
        quality = score_quality(bench, use_case.benchmark_weights)
        emoji = {"ok": "✅", "tight": "⚠", "no": "❌"}[fit.fit_status]
        tp_mark = "" if topo.tp_compatible else " (TP불가)"
        rows.append({
            "모델": model.display_name,
            "양자화": quant.upper(),
            "메모리": f"{emoji} {fit.total_used_gb:.0f}/{fit.total_available_gb:.0f}GB{tp_mark}",
            "최대 컨텍스트": fit.max_context_supported,
            "품질 점수": round(quality, 1) if quality > 0 else None,
            "라이선스": "✓상용" if model.license_commercial_ok else "✗비상용",
        })
    df = pd.DataFrame(rows)
    if not df.empty and "품질 점수" in df.columns:
        df = df.sort_values("품질 점수", ascending=False, na_position="last")
    return df


def build_matrix_panel() -> dict:
    from llm_gpu_fit.ui.wizard import gpu_model_choices, use_case_choices
    gr.Markdown("### 매트릭스 모드 — 모든 후보 비교")
    m_use_case = gr.Dropdown(label="용도", choices=use_case_choices(),
                             value="general")
    with gr.Row():
        m_gpu = gr.Dropdown(label="GPU 모델", choices=gpu_model_choices(),
                            value="h100_80")
        m_count = gr.Dropdown(label="GPU 수",
                              choices=[("1장", 1), ("2장", 2),
                                       ("4장", 4), ("8장", 8)],
                              value=1)
    m_btn = gr.Button("매트릭스 만들기", variant="secondary")
    m_table = gr.Dataframe(label="후보 모델 매트릭스", interactive=False, wrap=True)

    def _refresh(use_case, gpu_id, n):
        return build_matrix(use_case, gpu_id, int(n))

    m_btn.click(_refresh, inputs=[m_use_case, m_gpu, m_count], outputs=m_table)
    return {"use_case": m_use_case, "gpu": m_gpu, "count": m_count,
            "btn": m_btn, "table": m_table}
