from dataclasses import dataclass, field
from typing import Literal


GPUFormFactor = Literal["datacenter", "consumer", "workstation", "apple_silicon"]
Quantization = Literal["fp16", "bf16", "fp8", "int8", "int4", "awq", "gptq",
                       "q4_k_m", "q5_k_m", "q8_0"]
Framework = Literal["vllm", "sglang", "tgi", "llama.cpp", "mlx", "tensorrt-llm"]
Modality = Literal["text", "vision", "audio"]
Confidence = Literal["measured", "scaled", "theoretical"]


@dataclass(frozen=True)
class GPU:
    id: str
    name: str
    vendor: str
    vram_gb: float
    mem_bandwidth_gbs: float
    fp16_tflops: float
    int8_tops: float
    nvlink: bool
    form_factor: GPUFormFactor
    release_year: int = 0
    msrp_usd: float = 0
    cloud_hourly_usd: float = 0

    def total_vram(self, count: int = 1) -> float:
        return self.vram_gb * count

    def monthly_cloud_cost(self, count: int = 1) -> float:
        """24/7 운용 가정 — 클라우드 시간당 단가 × 730 × count."""
        return self.cloud_hourly_usd * 730 * count

    def amortized_monthly_owned(self, count: int = 1, years: int = 3) -> float:
        """자체 보유 시 N년 분할 가정 — MSRP / (years * 12) × count."""
        return self.msrp_usd / (years * 12) * count


@dataclass(frozen=True)
class Model:
    id: str
    display_name: str
    family: str
    params_total_b: float
    params_active_b: float
    context_window: int
    num_attention_heads: int
    num_kv_heads: int
    num_layers: int
    hidden_dim: int
    modalities: list[Modality]
    capabilities: list[str]
    license: str
    license_commercial_ok: bool
    hf_repo: str
    release_date: str = ""
    company: str = ""
    series: str = ""
    languages: tuple[str, ...] = ()
    # 인기도 1-5 (5=핫 트렌드, 3=일반, 1=레거시/니치)
    popularity_tier: int = 3

    def is_moe(self) -> bool:
        return self.params_active_b < self.params_total_b

    @property
    def company_or_family(self) -> str:
        return self.company or self.family.title()

    @property
    def series_or_family(self) -> str:
        return self.series or self.family.title()

    @property
    def supports_korean(self) -> bool:
        """한국어 네이티브이거나 다국어 지원 중 한국어 포함."""
        if "ko_native" in self.capabilities:
            return True
        return "ko" in self.languages or any(
            lang.startswith("multi-") for lang in self.languages
        )

    @property
    def korean_strength(self) -> str:
        """한국어 강도: 'native' / 'multilingual' / 'partial' / 'none'"""
        if "ko_native" in self.capabilities:
            return "native"
        if "ko" in self.languages:
            return "multilingual"
        if any(lang.startswith("multi-") for lang in self.languages):
            return "partial"
        return "none"


@dataclass
class UserInput:
    use_case: str
    gpu_id: str
    gpu_count: int
    commercial_required: bool
    korean_priority: bool
    onprem_required: bool
    tool_calling_required: bool
    concurrency: int = 1
    context_target: int = 8192
    quantization_preference: Quantization | None = None
    framework: Framework | None = None  # None=GPU 종류로 자동 선택


@dataclass
class MemoryFit:
    fits: bool
    fit_status: Literal["ok", "tight", "no"]
    weights_gb: float
    kv_cache_gb: float
    framework_overhead_gb: float
    total_used_gb: float
    total_available_gb: float
    max_context_supported: int
    max_concurrency_at_target_ctx: int


@dataclass
class TopologyCheck:
    tp_compatible: bool
    tp_reason: str
    nvlink_penalty: float
    recommended_tp: int
    warnings: list[str] = field(default_factory=list)


@dataclass
class Recommendation:
    model: Model
    quantization: Quantization
    framework: Framework
    memory: MemoryFit
    topology: TopologyCheck
    quality_score: float
    constraint_score: float
    final_score: float
    why: list[str]
    tradeoffs: list[str]
    measured_perf: dict | None = None
