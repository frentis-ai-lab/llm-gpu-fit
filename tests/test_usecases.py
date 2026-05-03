from llm_gpu_fit.core.usecases import load_use_cases, score_quality


def test_load_use_cases_includes_core_categories():
    ucs = {u.id: u for u in load_use_cases()}
    assert "general" in ucs
    assert "coding" in ucs
    assert "korean_general" in ucs
    assert "vision_ocr" in ucs
    assert "agent_tool" in ucs
    assert "long_context" in ucs


def test_coding_use_case_weights_sum_to_one():
    ucs = {u.id: u for u in load_use_cases()}
    total = sum(ucs["coding"].benchmark_weights.values())
    assert abs(total - 1.0) < 0.01


def test_score_quality_with_full_data():
    benchmark_scores = {
        "livecodebench": 64.2, "humaneval": 92.1,
        "bigcodebench": 55.0, "swe_bench_verified": 41.5,
    }
    weights = {
        "livecodebench": 0.4, "humaneval": 0.2,
        "bigcodebench": 0.2, "swe_bench_verified": 0.2,
    }
    score = score_quality(benchmark_scores, weights)
    expected = 64.2 * 0.4 + 92.1 * 0.2 + 55.0 * 0.2 + 41.5 * 0.2
    assert abs(score - expected) < 0.1


def test_score_quality_penalizes_missing_benchmarks():
    """단일 벤치마크만 있는 모델은 가중 평균에서 페널티."""
    benchmark_scores = {"humaneval": 90.0}
    weights = {"livecodebench": 0.4, "humaneval": 0.2, "swe_bench_verified": 0.4}
    score = score_quality(benchmark_scores, weights, min_coverage=0.0)
    # 0.2 * 90 = 18 (누락 항목은 0점 처리)
    assert abs(score - 18.0) < 0.01


def test_score_quality_returns_zero_when_below_min_coverage():
    benchmark_scores = {"humaneval": 90.0}
    weights = {"livecodebench": 0.4, "humaneval": 0.2, "swe_bench_verified": 0.4}
    # coverage=0.2, min_coverage=0.5 → 0 반환
    score = score_quality(benchmark_scores, weights, min_coverage=0.5)
    assert score == 0.0
