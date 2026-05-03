import gradio as gr

from llm_gpu_fit.core.data import load_gpus
from llm_gpu_fit.core.usecases import load_use_cases


def use_case_choices() -> list[tuple[str, str]]:
    return [(u.display_name_ko, u.id) for u in load_use_cases()]


def gpu_choices() -> list[tuple[str, str]]:
    out = []
    for g in load_gpus():
        for n in [1, 2, 4, 8]:
            label = f"{g.name} × {n}장 ({g.vram_gb*n:.0f}GB total)"
            out.append((label, f"{g.id}::{n}"))
    return out


def build_wizard() -> dict:
    gr.Markdown("## 무슨 모델을 쓰면 되나요?")

    use_case = gr.Dropdown(
        label="① 용도", choices=use_case_choices(),
        value="general", interactive=True,
    )
    gpu_selection = gr.Dropdown(
        label="② 우리 환경 (GPU 구성)", choices=gpu_choices(),
        value="h100_80::1", interactive=True,
    )

    gr.Markdown("### ③ 조직 제약")
    with gr.Row():
        commercial = gr.Checkbox(label="상용 라이선스 필요", value=False)
        korean = gr.Checkbox(label="한국어 우선", value=False)
        onprem = gr.Checkbox(label="온프레/폐쇄망", value=False)
        tool = gr.Checkbox(label="Tool calling 필수", value=False)

    with gr.Accordion("고급 옵션", open=False):
        context = gr.Slider(label="목표 컨텍스트 길이 (토큰)",
                            minimum=2048, maximum=131072,
                            value=8192, step=2048)
        concurrency = gr.Slider(label="동시 사용자 수",
                                minimum=1, maximum=64, value=1, step=1)
        quant_pref = gr.Dropdown(label="양자화 선호 (자동: 가장 큰 fit)",
                                 choices=[("자동", ""), ("BF16", "bf16"),
                                          ("FP8", "fp8"), ("INT8", "int8"),
                                          ("INT4", "int4"), ("AWQ", "awq"),
                                          ("GPTQ", "gptq")],
                                 value="")

    submit = gr.Button("추천 받기 →", variant="primary", size="lg")

    return {
        "use_case": use_case, "gpu_selection": gpu_selection,
        "commercial": commercial, "korean": korean,
        "onprem": onprem, "tool": tool,
        "context": context, "concurrency": concurrency,
        "quant_pref": quant_pref, "submit": submit,
    }
