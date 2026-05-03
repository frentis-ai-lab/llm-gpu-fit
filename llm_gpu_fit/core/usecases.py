from dataclasses import dataclass

import yaml

from llm_gpu_fit.core.data import DATA_DIR


@dataclass(frozen=True)
class UseCase:
    id: str
    display_name_ko: str
    display_name_en: str
    benchmark_weights: dict[str, float]
    required_capabilities: list[str]
    preferred_capabilities: list[str]


def load_use_cases() -> list[UseCase]:
    with (DATA_DIR / "use_cases.yaml").open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return [UseCase(**u) for u in raw["use_cases"]]


def score_quality(benchmark_scores: dict[str, float],
                  weights: dict[str, float],
                  min_coverage: float = 0.5) -> float:
    """벤치마크 가중 평균. 데이터가 없는 벤치마크는 0점으로 페널티.
    coverage가 min_coverage 미만이면 0 반환 (데이터 부족 모델 배제).
    """
    if not weights:
        return 0.0
    available = {k: w for k, w in weights.items() if k in benchmark_scores}
    if not available:
        return 0.0
    coverage = sum(available.values()) / sum(weights.values())
    if coverage < min_coverage:
        return 0.0
    # 누락 벤치마크는 0점 — 가중치 재정규화하지 않음
    score = sum(benchmark_scores[k] * w for k, w in available.items())
    return score / sum(weights.values())
