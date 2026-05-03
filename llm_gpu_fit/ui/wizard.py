import gradio as gr

from llm_gpu_fit.core.data import load_gpus
from llm_gpu_fit.core.usecases import load_use_cases


# 용도별 가이드: (권장 컨텍스트, 권장 동시성, 사용자에게 보여줄 설명)
_USE_CASE_GUIDE: dict[str, dict] = {
    "general": {
        "ctx_default": 8192, "ctx_min": 4096, "ctx_max": 16384,
        "conc_default": 4,
        "desc": "일상 대화·지식 질문. **8K-16K** 정도면 충분합니다. "
                "RAG로 긴 문서를 같이 넣을 거면 컨텍스트를 늘리세요.",
    },
    "reasoning": {
        "ctx_default": 16384, "ctx_min": 8192, "ctx_max": 32768,
        "conc_default": 1,
        "desc": "수학·추론. **16K-32K** 권장. 추론 모델(o1/R1 류)은 내부 추론 토큰을 많이 쓰므로 "
                "여유 있게. 동시 사용자 수는 적게(1-2) 잡는 편이 안전합니다.",
    },
    "coding": {
        "ctx_default": 32768, "ctx_min": 16384, "ctx_max": 65536,
        "conc_default": 4,
        "desc": "코딩 어시스턴트. **32K** 권장 (코드 컨텍스트 + 응답). "
                "대형 리포 분석은 64K, 단순 자동완성은 16K.",
    },
    "korean_general": {
        "ctx_default": 8192, "ctx_min": 4096, "ctx_max": 32768,
        "conc_default": 8,
        "desc": "한국어 일반 챗봇. **8K-16K**. 사내 다중 사용자가 많으면 동시성 8-16.",
    },
    "korean_reasoning": {
        "ctx_default": 16384, "ctx_min": 8192, "ctx_max": 32768,
        "conc_default": 2,
        "desc": "한국어 추론·분석. **16K**. 추론 토큰 여유 필요.",
    },
    "vision_ocr": {
        "ctx_default": 8192, "ctx_min": 4096, "ctx_max": 16384,
        "conc_default": 4,
        "desc": "OCR·문서 이해. **8K**가 보통. 이미지 토큰이 따로 들어가므로 텍스트 컨텍스트는 작아도 OK.",
    },
    "vision_general": {
        "ctx_default": 8192, "ctx_min": 4096, "ctx_max": 16384,
        "conc_default": 4,
        "desc": "비전 일반 (이미지 이해, VQA). **8K**. 이미지 + 짧은 응답 위주.",
    },
    "agent_tool": {
        "ctx_default": 32768, "ctx_min": 16384, "ctx_max": 65536,
        "conc_default": 2,
        "desc": "에이전트·Tool 호출. **32K** 권장 (대화 히스토리 + tool 결과 반복). "
                "복잡한 멀티 step agent면 64K.",
    },
    "long_context": {
        "ctx_default": 65536, "ctx_min": 32768, "ctx_max": 131072,
        "conc_default": 2,
        "desc": "긴 문맥·RAG. **64K-128K** 필수. 컨텍스트가 길수록 KV 캐시가 커져 동시 사용자 수가 줄어듭니다.",
    },
    "instruction_following": {
        "ctx_default": 8192, "ctx_min": 4096, "ctx_max": 16384,
        "conc_default": 4,
        "desc": "지시 이행 (요약·분류·변환). **8K**. 짧은 입출력 위주.",
    },
}


def use_case_choices() -> list[tuple[str, str]]:
    return [(u.display_name_ko, u.id) for u in load_use_cases()]


def gpu_model_choices() -> list[tuple[str, str]]:
    """GPU 모델만 (수량 분리). NVLink/form-factor 표기로 선택 도움."""
    out = []
    for g in load_gpus():
        nvlink = " · NVLink" if g.nvlink else ""
        ff = {
            "datacenter": "데이터센터", "consumer": "컨슈머",
            "workstation": "워크스테이션", "apple_silicon": "Apple Silicon",
        }[g.form_factor]
        label = f"{g.name} ({ff}{nvlink})"
        out.append((label, g.id))
    return out


def gpu_count_choices_for(gpu_id: str) -> list[tuple[str, int]]:
    """GPU 종류별로 가능한 수량 옵션. Apple Silicon은 1장만."""
    gpus = {g.id: g for g in load_gpus()}
    g = gpus.get(gpu_id)
    if g and g.form_factor == "apple_silicon":
        return [("1장", 1)]
    return [("1장", 1), ("2장", 2), ("4장", 4), ("8장", 8)]


def gpu_summary_text(gpu_id: str, count: int) -> str:
    gpus = {g.id: g for g in load_gpus()}
    g = gpus.get(gpu_id)
    if not g:
        return ""
    total = g.vram_gb * count
    eff = total * (0.75 if g.form_factor == "apple_silicon" else 1.0)
    eff_note = (f" · 추론 가용 약 **{eff:.0f}GB** (Metal 한계로 75%)"
                if g.form_factor == "apple_silicon"
                else f" · 추론 가용 **{total:.0f}GB**")
    nvlink_note = ""
    if count > 1:
        nvlink_note = (" · NVLink ✓ (분산 통신 빠름)" if g.nvlink
                       else " · ⚠ NVLink 없음 → PCIe 통신, 큰 모델일수록 페널티")
    return f"💾 총 VRAM **{total:.0f}GB**{eff_note}{nvlink_note}"


def use_case_guide_text(use_case_id: str) -> str:
    g = _USE_CASE_GUIDE.get(use_case_id)
    if not g:
        return ""
    return f"💡 **{g['desc']}**"


def build_wizard() -> dict:
    gr.Markdown("## 무슨 모델을 쓰면 되나요?\n세 가지만 고르면 추천이 나옵니다.")

    # === ① 용도 ===
    use_case = gr.Dropdown(
        label="① 어떤 용도로 쓰실 건가요?",
        choices=use_case_choices(),
        value="general", interactive=True,
    )
    use_case_guide = gr.Markdown(value=use_case_guide_text("general"))

    # === ② GPU (모델 + 수 분리) ===
    gr.Markdown("### ② 보유 GPU")
    with gr.Row():
        with gr.Column(scale=2):
            gpu_model = gr.Dropdown(
                label="GPU 모델",
                choices=gpu_model_choices(),
                value="h100_80", interactive=True,
            )
        with gr.Column(scale=1):
            gpu_count = gr.Dropdown(
                label="몇 장?",
                choices=[("1장", 1), ("2장", 2), ("4장", 4), ("8장", 8)],
                value=1, interactive=True,
            )
    gpu_summary = gr.Markdown(value=gpu_summary_text("h100_80", 1))

    # === ③ 조직 제약 ===
    gr.Markdown("### ③ 조직·서비스 제약")
    with gr.Row():
        commercial = gr.Checkbox(label="상용 라이선스 필요", value=False,
                                 info="제품·서비스로 판매할 거면 체크")
        korean = gr.Checkbox(label="한국어 우선", value=False,
                             info="한국어 능력 가산점")
        onprem = gr.Checkbox(label="온프레/폐쇄망", value=False,
                             info="외부 API 차단 환경")
        tool = gr.Checkbox(label="Tool calling 필수", value=False,
                           info="Function calling으로 외부 도구 호출")

    # === 고급 옵션 ===
    with gr.Accordion("컨텍스트·동시 사용자 (고급)", open=False):
        context = gr.Slider(
            label="목표 컨텍스트 길이 (토큰)",
            minimum=2048, maximum=131072, value=8192, step=2048,
            info="모델에 한 번에 넣는 입출력의 최대 길이",
        )
        gr.Markdown(
            "용도를 바꾸면 권장값이 자동 적용됩니다. "
            "토큰은 한국어 약 1글자 = 1.5토큰, 영어 단어 ≈ 1.3토큰."
        )

        concurrency = gr.Slider(
            label="동시 사용자 수",
            minimum=1, maximum=64, value=1, step=1,
            info="같은 시간에 동시에 요청을 보낼 사용자 수",
        )
        gr.Markdown(
            "1인 PoC면 1, 사내 챗봇이면 4-16, 고객 서비스면 32+. "
            "동시성이 늘면 KV 캐시가 곱해져 메모리가 더 필요합니다."
        )

        framework = gr.Dropdown(
            label="서빙 프레임워크",
            choices=[
                ("자동 (NVIDIA→vLLM, Apple→MLX)", ""),
                ("vLLM (프로덕션 표준, 고성능 배칭)", "vllm"),
                ("SGLang (vLLM 대안, JSON/tool 강점)", "sglang"),
                ("TensorRT-LLM (NVIDIA 최적화, H100/H200)", "tensorrt-llm"),
                ("Ollama / LM Studio / llama.cpp (GGUF, 개인용)", "llama.cpp"),
                ("MLX (Apple Silicon 전용)", "mlx"),
            ],
            value="",
            info="vLLM/SGLang은 AWQ/GPTQ/INT4 사용. Ollama·LM Studio는 GGUF만(Q4_K_M 등). "
                 "프로덕션·다중 사용자는 vLLM, 노트북·1인 사용은 Ollama 권장.",
        )

        quant_pref = gr.Dropdown(
            label="양자화 선호",
            choices=[
                ("자동 (BF16 시도 → 안 되면 양자화 fallback)", ""),
                ("AWQ — INT4 권장 (vLLM/SGLang 표준)", "awq"),
                ("INT4 — GPU 추론 INT4 일반", "int4"),
                ("GPTQ — INT4 대안", "gptq"),
                ("Q4_K_M — Ollama/llama.cpp INT4 표준", "q4_k_m"),
                ("Q5_K_M — 약간 더 품질 좋은 GGUF", "q5_k_m"),
                ("Q8_0 — GGUF INT8", "q8_0"),
                ("FP8 — H100/H200 네이티브", "fp8"),
                ("INT8 — 보수적", "int8"),
                ("BF16 — 양자화 없음, 품질 최고", "bf16"),
            ],
            value="awq",
            info="대부분 INT4 계열(AWQ/GPTQ/Q4_K_M)을 씁니다. 메모리 절감 4배, 품질 손실 5-10%.",
        )

    submit = gr.Button("추천 받기 →", variant="primary", size="lg")

    # === 콜백: 용도 변경 시 가이드와 컨텍스트·동시성 자동 갱신 ===
    def _on_use_case_change(uc_id):
        g = _USE_CASE_GUIDE.get(uc_id, {})
        return (
            use_case_guide_text(uc_id),
            gr.update(value=g.get("ctx_default", 8192)),
            gr.update(value=g.get("conc_default", 1)),
        )

    use_case.change(
        _on_use_case_change,
        inputs=[use_case],
        outputs=[use_case_guide, context, concurrency],
    )

    # === 콜백: GPU 모델·수 변경 시 요약 갱신 + Apple Silicon 1장 강제 ===
    def _on_gpu_model_change(gpu_id, count):
        choices = gpu_count_choices_for(gpu_id)
        valid_counts = [c[1] for c in choices]
        new_count = count if count in valid_counts else valid_counts[0]
        return (
            gr.update(choices=choices, value=new_count),
            gpu_summary_text(gpu_id, new_count),
        )

    def _on_gpu_count_change(gpu_id, count):
        return gpu_summary_text(gpu_id, count)

    gpu_model.change(
        _on_gpu_model_change, inputs=[gpu_model, gpu_count],
        outputs=[gpu_count, gpu_summary],
    )
    gpu_count.change(
        _on_gpu_count_change, inputs=[gpu_model, gpu_count],
        outputs=[gpu_summary],
    )

    return {
        "use_case": use_case,
        "gpu_model": gpu_model,
        "gpu_count": gpu_count,
        "commercial": commercial, "korean": korean,
        "onprem": onprem, "tool": tool,
        "context": context, "concurrency": concurrency,
        "quant_pref": quant_pref, "framework": framework,
        "submit": submit,
    }
