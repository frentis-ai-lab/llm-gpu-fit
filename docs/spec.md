---
title: LLM-GPU-Fit — GPU 구성별 추론 모델 추천 계산기
date: 2026-05-04
status: design-approved
owner: andy
codename: falcon (project)
---

# LLM-GPU-Fit 설계서

## 1. 배경과 목적

KB손해보험 컨설팅의 유엔진 등 컨소시엄 파트너, 그리고 SKT/KOSTA 교육 수강생들이 반복적으로 같은 형태의 질문을 한다.

> "[목적]을 하려고 하는데, [조직]에서 [GPU 구성]을 쓰면 무슨 모델 쓰면 되나요?"

현재는 매번 수동으로 답하고, 답변도 "이거 쓰세요" 한 줄에 그쳐 의사결정 근거가 약하다. 이 문제를 셀프 서비스 도구로 해소한다. 도구는 HuggingFace Spaces에 공개해 frentis 브랜드 자산으로도 활용한다.

## 2. 범위

**포함**
- 추론 전용 (학습/파인튜닝 제외)
- **메모리(VRAM) 적합성 중심** — "이 모델이 이 GPU 구성에 들어가는가, 컨텍스트 얼마까지 가능한가"
- 핵심 오픈웨이트 모델 + 한국어 특화 모델
- NVIDIA 데이터센터/컨슈머 GPU + Apple Silicon
- 양자화: FP16/BF16/FP8/INT8/INT4/AWQ/GPTQ/GGUF 주요 변형
- 기능별 카테고리 (코딩, 추론, OCR/비전, Agent/Tool, 한국어, 긴 문맥 등)
- 품질 벤치마크 (자동 수집 가능)

**제외**
- 파인튜닝/사전학습 메모리 계산
- AMD/Intel Gaudi (한국 엔터프라이즈 비주류)
- 사용자 제보 시스템 (운영 비용 대비 유입 부족)
- **GPU별 정확한 throughput 예측** — 실측 데이터가 공개적으로 모든 조합에 존재하지 않음. TPS/TTFT는 *알려진 경우에만 부가정보로* 표시하고 추천 점수에는 반영하지 않는다.

## 3. 핵심 사용자 경험

### 3.1 진입: 3-필드 위저드

```
① 용도         [▾ 코딩 어시스턴트                       ]
② 우리 환경    [▾ H100 80GB × 1장                     ]
③ 조직 제약    [☐ 상용 라이선스 필요  ☐ 한국어 우선
                 ☐ 온프레/폐쇄망     ☐ Tool calling 필수]

[ 고급 옵션 ▾ ]  컨텍스트, 동시 사용자 수, 양자화 선호

                 [ 추천 받기 → ]
```

### 3.2 결과: "왜 이 모델인가"까지

추천 모델 1순위 카드에 다음을 표시한다.
- 모델명, 양자화, **들어가는 컨텍스트 최대치**
- **VRAM 사용량/여유** (메인 신호), 동시성 가정 시 KV 캐시 한계
- 성능 정보(TPS/TTFT)는 *공식·MLPerf·vLLM perf 등 실측이 있을 때만* 표시. 없으면 "성능 데이터 없음, 메모리 적합성만 검증" 명시
- ✅ 추천 사유 (벤치마크 점수 + 용도 적합도 + 라이선스 + 메모리 적합성)
- ⚠ 알아둘 트레이드오프
- 2·3순위 모델 카드 (요약형)
- `[ 비교 매트릭스 열기 ]` 버튼

### 3.3 보조 모드

- **매트릭스 모드** — 모든 후보 모델을 표로 표시. 컬럼은 용도별 벤치마크 + capability 배지. 엔지니어/심층 비교용
- **자주 묻는 시나리오 프리셋** — 홈에 카드 4-6개 ("H100 1장으로 코딩", "RTX 4090 4장으로 한국어 RAG", "맥북 M3 Ultra로 사내 챗봇" 등)
- **공유 URL** — `?use=coding&gpu=h100x1&ko=1&commercial=1` 형태, 한 줄 답변용

## 4. 데이터 모델

### 4.1 Use Case 카탈로그

| Use Case | 가중 벤치마크 | 필수 capability |
|----------|------------|----------------|
| 일반 대화/지식 | MMLU-Pro, GPQA, MT-Bench, IFEval, Arena Elo | text |
| 수학/추론 | MATH, GSM8K, AIME, GPQA-Diamond | text |
| 코딩 | LiveCodeBench, HumanEval, BigCodeBench, SWE-bench Verified | text, long-ctx, JSON |
| 한국어 일반 | KMMLU, HAE-RAE, LogicKor, Open Ko-LLM avg | ko |
| 한국어 추론 | LogicKor 추론, Ko-IFEval | ko |
| OCR/문서이해 | OCRBench, DocVQA, ChartQA, AI2D | vision |
| 비전 일반 | MMMU, MMMU-Pro, MathVista, RealWorldQA | vision |
| Agent/Tool use | BFCL v3, τ-bench, ToolBench, IFEval | tool, JSON |
| 긴 문맥/RAG | RULER, LongBench v2, NIAH | ≥128K ctx |
| 지시 이행 | IFEval, MT-Bench | text |

### 4.2 테이블 스키마

**`models`** — `id, family, display_name, params_total, params_active, context_window, modalities[], capabilities[], license, license_commercial_ok, release_date, hf_repo, source_url`

**`benchmarks`** (long format) — `model_id, benchmark_id, score, max_score, normalized_0_100, measured_by, measured_at, source_url, confidence`

**`gpus`** — `id, vendor, name, vram_gb, mem_bandwidth_gbs, fp16_tflops, int8_tops, nvlink_bw, form_factor, release_year, msrp_usd`

**`inference_perf`** — `model_id, gpu_id, gpu_count, quantization, framework, context_used, batch_size, concurrency, tps_single, tps_aggregate, ttft_ms, itl_ms, vram_used_gb, kv_cache_gb, source, measured_at, confidence`

**`use_cases`** — `id, display_name_ko, display_name_en, benchmark_weights, required_capabilities[], preferred_capabilities[]`

### 4.3 Capability 배지

`👁 vision` `🔧 tool` `📋 json` `🧠 reasoning` `🇰🇷 ko-native` `📜 1M ctx` `⚖ commercial-ok`

## 5. 계산 엔진

### 5.1 VRAM

```
weights_gb       = params_total × bytes_per_param(quant) × 1.05
activations_gb   = weights_gb × 0.05~0.10  (framework 흡수)
kv_cache_gb      = batch × seq_len × layers × 2 × kv_heads × head_dim × bytes(kv_quant) / 1e9
framework_overhead_gb = vLLM/SGLang 1.5 / llama.cpp 0.5 / TensorRT-LLM 2.0

total_vram = weights + activations + kv_cache + framework_overhead
fits = total_vram ≤ Σ(gpu_vram) × 0.95
```

bytes_per_param: FP16/BF16=2.0, FP8=1.0, INT8=1.0, INT4/AWQ/GPTQ=0.5, Q4_K_M=0.55, Q5_K_M=0.65, Q8_0=1.05

다중 GPU: tensor parallel로 합산 가능, NVLink 없으면 TPS ×0.85.

### 5.2 메모리 적합성 (1차 신호)

VRAM 적합 여부와 가능 컨텍스트 길이를 핵심 출력으로 삼는다.

```
fits_at_ctx(ctx) = total_vram(ctx, batch, concurrency) ≤ Σ(gpu_vram) × 0.95

max_context_supported = arg max ctx s.t. fits_at_ctx(ctx) AND ctx ≤ model.context_window
max_concurrency       = floor((vram_total × 0.95 − weights − fw_overhead) / kv_cache_per_request)
```

추천 카드에는 다음을 표시한다.
- "✅ 들어감 / ⚠ 빠듯 / ❌ 안 들어감"
- 들어가는 최대 컨텍스트 (예: "최대 64K까지")
- 동시성 N에서의 KV 캐시 한계 (예: "동시 8 요청 시 최대 16K 컨텍스트")

빠듯 기준: VRAM 사용 85-95%. 95% 초과면 OOM 위험으로 ❌.

### 5.2.1 다중 GPU 토폴로지 (실무에서 가장 자주 깨지는 지점)

VRAM 합산만으로는 다중 GPU 시나리오를 제대로 평가할 수 없다. "4090×4(96GB)에 70B INT4 들어가는데 왜 1장 H100의 32B보다 느린가" 같은 질문이 핵심.

**검사 항목 (모두 추천 카드에 명시)**:

1. **TP(Tensor Parallel) 가능 여부** — `model.num_attention_heads % gpu_count == 0` AND `model.num_kv_heads % gpu_count == 0`
   - 예: Llama 3 70B(heads=64, kv=8) → TP 1/2/4/8 가능, TP 3/5/6 불가
   - 예: Qwen3-32B(heads=64, kv=8) → 동일
   - 불가하면 "❌ 이 GPU 수로 TP 분산 불가, GPU 수 변경 필요" 표시

2. **인터커넥트 종류**
   - 데이터센터 GPU (H100/A100/H200/B200): NVLink/NVSwitch → all-reduce 빠름
   - 컨슈머 GPU (RTX 4090/5090/3090): **NVLink 없음** → PCIe 4.0/5.0 only
   - PCIe 통신 페널티 경고 표시: "⚠ 4090×N은 NVLink 없음 → all-reduce가 PCIe 통과, 70B+ 모델에서 TPS 30-50% 저하 보고"

3. **메모리 분산 vs 활용도 트레이드오프**
   - GPU 수가 늘면: weights 분산 ✓ / KV 캐시는 통상 sequence-parallel 또는 단일 복제 / framework overhead는 GPU당 발생 → N배
   - 권장 토폴로지 자동 계산: "이 모델은 TP=2가 최적 (TP=4는 통신 오버헤드 ~25% 추정)"
   - 휴리스틱: weights가 1장 VRAM의 60% 이하로 들어가면 TP 늘리기보다 1장 권장

4. **"더 작은 모델 1장이 더 낫지 않나?" 비교 위젯**
   - 결과 화면 하단에 항상: 같은 GPU의 1장 구성에서 가능한 한 단계 작은 모델을 자동 추천하여 비교
   - 예: 입력이 "4090×4 + 70B INT4"면 "대안: 4090×1 + 32B BF16 → 같은 용도 점수 92 vs 95, TPS는 ~3배 빠름 추정"

5. **MoE 모델의 EP/TP 혼합**
   - DeepSeek-V3, Mixtral 등은 expert parallel이 효율적 → "이 모델은 EP=N 권장, vLLM/SGLang 옵션 명시"

**카드 표시 예**:

```
H100 80GB × 1장 · Qwen3-32B BF16 · 32K ctx
  ✅ 메모리 적합 · 토폴로지 단순 (TP 불필요)
  여유 16GB · NVLink 무관

vs

RTX 4090 × 4장 · Llama 3.3 70B INT4 · 32K ctx
  ⚠ 메모리 빠듯 (사용 88%)
  ⚠ NVLink 없음 → PCIe 통신, all-reduce 30-50% 페널티 추정
  💡 대안: RTX 4090 × 1장 + Qwen3-32B INT4 → 품질 비슷, 안정성 ↑
```

### 5.3 Throughput (선택적, 실측 있을 때만)

`inference_perf`에 매칭되는 행이 있으면 표시한다. 없으면 카드에 "성능 데이터 없음" 라벨만 두고 추정값은 노출하지 않는다 (잘못된 기대치 방지).

매칭 우선순위:
1. (model, gpu, gpu_count, quant, framework) 정확 일치 (🟢 측정)
2. 같은 모델·동일 GPU·다른 quant — 참고치로 표시, "다른 양자화 측정값" 라벨 (🟡 참고)

같은 family 다른 모델로의 외삽이나 roofline 이론 추정은 **표시하지 않는다**. 실측이 없으면 그냥 비워둔다.

### 5.4 신뢰도 (메모리 vs 성능)

- **메모리 계산**: 결정론적 공식, 항상 🟢 (오차 ±10% 명시)
- **성능 표시**: 🟢 측정 / 🟡 다른 quant 참고 / 데이터 없음(빈 칸)

### 5.5 추천 점수

성능 절댓값을 추천에 반영하지 않는다. 메모리·토폴로지는 게이트와 페널티로 작용.

```
1차 게이트: 메모리·토폴로지 적합성
   - ❌ 메모리 안 들어감          → 추천에서 제외
   - ❌ TP 분산 불가 (heads 미정합) → 제외
   - ⚠ 빠듯 (VRAM 85-95%)         → 적합도 ×0.7
   - ✅ 적합                       → 적합도 ×1.0
   - required_capabilities 미충족   → 제외

2차 점수:
   적합도 = 0.65 × 품질점수(use_case 가중)
         + 0.35 × 제약충족(boolean → 0 or 1)
         − 토폴로지 페널티
         − 기타 페널티

토폴로지 페널티
   - 컨슈머 GPU 다중(NVLink 없음) + 모델 크기 ≥ 30B: −15%
   - 컨슈머 GPU 다중(NVLink 없음) + 모델 크기 ≥ 70B: −25%
   - GPU 수가 권장 토폴로지보다 많아 통신 오버헤드 추정 ≥ 20%: −10%
```

품질 가중을 높인 이유: 메모리는 통과/탈락 게이트로 옮겼고, 가용 후보 중에서는 품질이 사실상 가장 큰 변별 요인이기 때문.

제약충족 = 체크된 조건 만족 비율 (라이선스, 한국어, 폐쇄망, tool calling 등)

페널티
- 품질 데이터 6개월 이상 + 새 버전 출시: −10%
- 동일 family 내 더 좋은 모델이 같은 GPU에서 더 잘 돌면: −15%
  ("동일 family" = 같은 vendor + 같은 base 아키텍처 계열, 예: Llama 3.1/3.2/3.3, Qwen 2.5/3, DeepSeek V2.5/V3/V3.1)
- EOL/공식 deprecated: −30%

**Use Case의 필수 capability 미충족 모델 처리**: `required_capabilities`를 모두 갖추지 못한 모델은 추천 후보에서 제외 (적합도 0). 매트릭스 모드에서는 회색 처리하여 표시하되 "비전 모델 아님" 같은 사유 라벨 부착. `preferred_capabilities`는 적합도 가산점 +5%.

체크박스 반응
- "상용 라이선스 필요" → 비상용 모델 적합도 0
- "한국어 우선" → 한국어 보정 가중 ×2
- "온프레/폐쇄망" → 클라우드 전용 모델 제외

### 5.5 엣지 케이스

- **MoE**: weights = total params로 계산, TPS = active params 기준 roofline
- **Reasoning 모델**: TPS에 "+ 추론 토큰" 주석, context에 reasoning budget 별도
- **Vision**: 이미지 토큰 (≈ image_size / patch²) 추가 계산
- **Apple Silicon**: vram = system_ram × 0.75 (Metal), framework는 mlx/llama.cpp
- **Long context**: KV cache가 weights 추월하는 임계점 자동 표시

## 6. 아키텍처

```
┌─────────────────────────────────────────────────┐
│  HF Space: frentis/llm-gpu-fit (Gradio)         │
│  - 위저드 / 결과 / 매트릭스 / 프리셋             │
│  - 계산 엔진 (Python, parquet 인메모리/DuckDB)  │
└──────────┬──────────────────────────────────────┘
           │ reads parquet
           ▼
┌─────────────────────────────────────────────────┐
│  HF Dataset: frentis/llm-gpu-fit-data           │
│  - models / benchmarks / gpus / inference_perf  │
│  - use_cases (정적 YAML, 레포 내 관리)          │
└──────────▲──────────────────────────────────────┘
           │ writes weekly
┌──────────┴──────────────────────────────────────┐
│  GitHub Actions (cron, Python)                  │
│  collectors/                                    │
│  필수 (메모리·품질 중심)                        │
│  ├─ huggingface_hub.py     모델 메타 + LLB v2   │
│  ├─ open_ko_llm.py          한국어 리더보드     │
│  ├─ logickor.py                                 │
│  ├─ kmmlu_haerae.py                             │
│  ├─ ollama_library.py       quant 메타 참고치   │
│  └─ model_card_parser.py    HF 모델카드 capability·context 추출
│  선택 (성능 데이터, 있으면 표시만)              │
│  ├─ mlperf_inference.py     반기 CSV            │
│  └─ vllm_perf.py            vLLM 공식 perf      │
└─────────────────────────────────────────────────┘
```

수동 작업: 월 1회 30분, Reddit r/LocalLLaMA / HF Forums 주간 요약에서 RTX/Apple Silicon 실측 데이터를 큐잉해 frentis 측 검토 후 dataset PR.

## 7. 기술 스택

- UI: **Gradio 5.x** (HF Spaces 네이티브)
- 데이터: **HF Datasets (parquet)** + **DuckDB** (Space 내 쿼리)
- 수집: **GitHub Actions (Python 3.13, uv)** + cron `weekly`
- 라이브러리: `huggingface_hub`, `pandas`, `duckdb`, `pyarrow`, `httpx`, `beautifulsoup4`
- 언어: 한국어/영어 토글
- 네임스페이스: `frentis/llm-gpu-fit` (변경 가능)

## 8. 라이선스 처리

- 데이터셋 자체는 Apache 2.0 (frentis 가공물)
- 각 벤치마크/실측 출처는 row-level `source_url`로 표기
- LMSYS Arena Elo는 비상업 조건 → 캐시·표시만, 재배포 없는 형태
- NVIDIA NIM 표는 출처 표기 후 인용

## 9. 마일스톤 (제안, plan 단계에서 확정)

1. **MVP** (1-2주): 위저드 + 결과 카드 + 이론 계산만, 모델 20개 + GPU 10종 시드 데이터
2. **데이터 자동화** (1주): GitHub Actions cron, HF Datasets 연동, 핵심 collector 5종
3. **매트릭스/필터** (1주): 보조 모드, 프리셋 카드
4. **공개 베타** (1주): HF Spaces 배포, frentis-site 임베드, 피드백 수렴

## 10. 검증

- VRAM 계산: HuggingFace `accelerate estimate-memory` 결과와 ±10% 일치
- 알려진 한계 케이스: "Llama 3.3 70B BF16을 H100 1장에서" → ❌ (140GB > 80GB), INT4면 ✅
- 한국어 케이스: "Solar Pro 22B BF16을 RTX 4090 1장에서" → ❌ (44GB > 24GB), INT4면 ✅
- 컨텍스트 한계: "Qwen3-32B INT4를 H100 1장에서 32K vs 128K" → KV 캐시 차이 검증

## 11. 비범위/유보

- 파인튜닝 메모리 (별도 spec 가능)
- 임베딩/리랭크 모델 (별도 카탈로그)
- 가격(Cost) 차원 — 추후 확장
- 사용자 제보 — 운영 부담 대비 유입 부족, 보류
