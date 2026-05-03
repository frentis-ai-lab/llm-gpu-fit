---
title: LLM GPU Fit
emoji: 🧮
colorFrom: indigo
colorTo: pink
sdk: gradio
sdk_version: 5.0.0
app_file: app.py
pinned: false
license: apache-2.0
---

# LLM GPU Fit 🧮

> "이 GPU 구성에 어떤 LLM이 들어가나, 어떤 모델을 쓰면 되나" 셀프 서비스 계산기.

**Live**: https://huggingface.co/spaces/frentis/llm-gpu-fit

## 무엇을 답하는가

세 가지 입력만으로 추천을 받습니다.

1. **용도** — 코딩 / 한국어 / OCR / Agent / 추론 / 일반 등 10종
2. **GPU 구성** — H100 1장, RTX 4090 4장, M3 Ultra 등
3. **조직 제약** — 상용 라이선스, 한국어 우선, 온프레, Tool calling

추천은 1순위 카드와 함께 **왜 이 모델인지** + **트레이드오프** + **2·3순위 대안** + **"더 작은 모델 1장이 낫지 않나?" 비교**까지 한 화면에 보여줍니다.

## 무엇을 검증하는가

- ✅ **메모리 적합성** — weights + KV cache + framework overhead가 GPU에 들어가는지
- ✅ **토폴로지** — TP heads 정합, NVLink 유무에 따른 통신 페널티
- ✅ **품질 벤치마크** — 용도별 가중치 (LiveCodeBench, MMLU-Pro, KMMLU, RULER 등)
- ✅ **라이선스/제약** — 상용 가능 여부, 한국어 네이티브, Tool calling 지원

## 자세한 설계

[docs/spec.md](docs/spec.md) — 데이터 모델, 계산식, 토폴로지 검사, 추천 점수 가중

## 기여

벤치마크/모델 추가 PR 환영합니다. `data/seed/`의 형식을 따라 `models.parquet` 또는 `benchmarks.parquet`에 행 추가.

## 라이선스

Apache 2.0
