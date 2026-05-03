from llm_gpu_fit.core.types import Recommendation


def render_recommendation_card(rec: Recommendation, rank: int) -> str:
    medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, str(rank))
    fit_emoji = {"ok": "✅", "tight": "⚠", "no": "❌"}[rec.memory.fit_status]

    why_md = "\n".join(f"- {w}" for w in rec.why) if rec.why else "- (벤치마크 데이터 부족)"
    tradeoffs_md = "\n".join(f"- {t}" for t in rec.tradeoffs) if rec.tradeoffs else "- 없음"

    return f"""
### {medal} {rank}순위: {rec.model.display_name}

**구성**: {rec.quantization.upper()} · {rec.framework} · 컨텍스트 {rec.memory.max_context_supported:,} 토큰까지

**메모리**: {fit_emoji} {rec.memory.total_used_gb:.1f}GB / {rec.memory.total_available_gb:.1f}GB 사용
(weights {rec.memory.weights_gb:.1f}GB · KV {rec.memory.kv_cache_gb:.1f}GB · framework {rec.memory.framework_overhead_gb:.1f}GB)

**최대 동시성**: 목표 컨텍스트 기준 {rec.memory.max_concurrency_at_target_ctx} 요청

**점수**: 품질 {rec.quality_score:.1f}/100 · 제약 충족 {rec.constraint_score:.0f}% · **종합 {rec.final_score:.1f}**

#### ✅ 왜 이 모델인가
{why_md}

#### ⚠ 알아둘 것
{tradeoffs_md}

---
"""


def render_results(recs: list[Recommendation]) -> str:
    if not recs:
        return ("### 😅 추천할 모델이 없습니다\n\n"
                "이 GPU 구성에 들어가는 모델이 없거나 모든 제약을 만족하는 모델이 없습니다. "
                "GPU 수를 늘리거나 제약 체크를 줄여보세요.")
    cards = [render_recommendation_card(r, i + 1) for i, r in enumerate(recs)]
    return "\n".join(cards)


def render_alternative_section(top: Recommendation,
                               alt: Recommendation | None) -> str:
    if alt is None:
        return ""
    delta = alt.quality_score - top.quality_score
    delta_str = f"{delta:+.1f}점"
    return f"""
### 💡 대안: 1장 구성으로 더 단순하게

같은 GPU **1장**에서 가능한 추천: **{alt.model.display_name}** ({alt.quantization.upper()})

| 항목 | 다중 GPU 추천 | 1장 대안 |
|------|------------|--------|
| 모델 | {top.model.display_name} | {alt.model.display_name} |
| 양자화 | {top.quantization.upper()} | {alt.quantization.upper()} |
| 품질 점수 | {top.quality_score:.1f} | {alt.quality_score:.1f} ({delta_str}) |
| 메모리 사용 | {top.memory.total_used_gb:.0f}GB / {top.memory.total_available_gb:.0f}GB | {alt.memory.total_used_gb:.0f}GB / {alt.memory.total_available_gb:.0f}GB |
| 토폴로지 페널티 | {top.topology.nvlink_penalty*100:.0f}% | 0% |

품질 차이가 작고 운영 단순성·통신 오버헤드를 고려하면 1장 구성이 나을 수 있습니다.

---
"""
