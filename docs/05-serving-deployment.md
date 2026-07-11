# 서빙·배포 설계서 (Serving & Deployment) — 문서 B

> **코드잇 스프린트 AI 엔지니어링 12기 · 초급 프로젝트 · 3팀 (ULTRA CAPSHYONG ITEM WITH 4 VALUES)**
> 문서 계보: 계획서 → SRS → PRD → 실행 전략 → A(데이터 모델) → **〔보강 B〕 서빙·배포 설계** (C: UX·데모는 별도)
> 작성: 이형기(Model Architect) · 서빙 트랙 owner 최중열 협의 · 작성일 2026-06-26 · v0.1
> 충족 요구사항: NFR-03(추론 속도), FR-07(후처리), FR-11/12(서빙·재학습), NFR-07(확장성) · 실행전략 §5②·§6.2-3

---

## 0. 문서 개요

문서 A가 *데이터가 어떻게 생겼는가*를 정의했다면, 본 문서는 **그 데이터를 얹은 모델이 어떻게 웹에서 동작·배포·관측·개선되는가**를 정의한다. 중심은 세 가지.
1. **이중 트랙의 서빙면 구현** — 서빙에는 앙상블이 아니라 **단일 경량 모델**(전략 §5②)
2. **데이터 모델 → API 연결** — `drug_master`·`class_map.json`을 응답 스키마로 (문서 A)
3. **데모 안정성** — 라이브 발표가 깨지지 않도록 런북·워밍업·폴백 (§8)

> 범위 제외: 화면 디자인·의료 디스클레이머·발표 스크립트(→문서 C), 데이터 사전(→문서 A).

---

## 1. 서빙 아키텍처 토폴로지

전략 §6.2의 티어를 배포 형태로 구체화한다. **T1을 발표 보장선으로** 확보하고, 여력에 따라 T2.

### T1 (MVP·필수) — Gradio @ Hugging Face Spaces
```
[사용자 브라우저]
      │ 이미지 업로드
      ▼
┌─────────────────────────────────────────┐
│  Hugging Face Space (Gradio)             │
│  app.py                                  │
│   ├─ serving model (.pt, 단일 YOLO)       │
│   ├─ class_map.json   (model_idx↔cat_id) │  ← 문서 A SSOT
│   ├─ drug_master.json (cat_id↔약정보)     │  ← 문서 A SSOT
│   └─ predict() → 박스 오버레이 + 약 카드    │
└─────────────────────────────────────────┘
```
- 장점: 무료·공유 URL·웹 UI 자동 제공 → **가장 빠른 라이브 데모**.
- 유의: 휴면 후 **콜드 스타트**, 무료 CPU 추론 → §8 워밍업·§7 지연 예산으로 대응.

### T2 (심화) — FastAPI 백엔드 + Next.js 프론트
```
[사용자 브라우저]
      │
      ▼
[Next.js (Vercel)] ──REST──▶ [FastAPI (Docker / Render·HF Docker)]
                                ├─ /predict /drugs /health /feedback
                                ├─ serving model + class_map + drug_master
                                └──▶ [Supabase(Postgres): prediction/_item/feedback]
```
- 정식 웹앱 형태. 프론트=Vercel(기존 AIMAP 스택), 추론=컨테이너, 로그=Supabase(문서 A 테이블).

---

## 2. 모델 패키징 (Serving Bundle)

서빙 트랙 산출물을 **자기완결 번들**로 고정 → `model_registry`(문서 A)와 1:1.

```
serving_bundle/<model_version>/
├── model/
│   ├── weights.pt            # 단일 YOLO (서빙 트랙). 필요 시 model.onnx
│   └── class_map.json        # model_index ↔ category_id  ★SSOT
├── data/
│   └── drug_master.json      # category_id ↔ 약 정보       ★SSOT
├── config/
│   └── inference.yaml        # input_size, score_thr, nms_iou, max_det=4
└── meta/
    └── model_card.json       # model_version, val_mAP, input_resolution, trained_at
```

| 결정 | 권장 |
| --- | --- |
| 포맷 | **T1=Ultralytics `.pt` 직접**(가장 단순) / T2=선택적 **ONNX** export(지연 최적화) |
| 버전 네이밍 | `model_registry.model_version`과 동일 (예: `yolo11s_r1024_v3`) |
| SSOT 동봉 | `class_map.json`·`drug_master.json`을 번들에 포함 → 모델·정보 버전 불일치 차단 |

> 서빙 모델은 **대회 앙상블과 분리**. 무거운 앙상블은 배포하지 않는다(NFR-03).

---

## 3. 추론 파이프라인 (서빙)

```
[multipart 업로드]
  → [검증: 타입(png/jpg)·크기(≤10MB)]
  → [디코드 (PIL/cv2)]
  → [전처리: resize(model input), normalize]
  → [추론: 단일 YOLO]
  → [후처리: NMS · score_thr · max_det=4]
  → [매핑: model_index → category_id  (class_map.json)]
  → [조인: category_id → 약 카드      (drug_master.json)]
  → [응답: 박스 오버레이(옵션) + JSON]
  → [로깅: prediction/_item  (T2, 비동기)]   ← 응답 지연에 미포함
```

- **불변식(문서 A §2):** 출력 `category_id` ∈ Test 40 ∧ ∈ `drug_master`. 미존재 시 fallback 카드(`category_id=-1`).
- 로깅은 응답 후 비동기 → 사용자 체감 지연에 영향 없음.

---

## 4. API 명세

> 응답 스키마는 문서 A의 `prediction / prediction_item / drug_master`와 직접 매핑.

### 4.1 `POST /predict`
- **요청:** `multipart/form-data`, field `image` (png/jpg, ≤10MB)
- **응답 200:**
```json
{
  "prediction_id": "uuid",
  "model_version": "yolo11s_r1024_v3",
  "latency_ms": 234,
  "num_detections": 3,
  "detections": [
    {
      "item_id": "uuid",
      "category_id": 24,
      "bbox": { "x": 156, "y": 247, "w": 211, "h": 456 },
      "score": 0.91,
      "rank": 1,
      "drug": {
        "dl_name": "제품명",
        "dl_material": "성분명",
        "di_class_no": "약품 분류",
        "di_etc_otc_code": "일반의약품",
        "drug_shape": "원형",
        "color_class1": "하양",
        "print_front": "각인",
        "dl_company": "제조사",
        "repr_image": "https://.../repr.png"
      }
    }
  ]
}
```
- **에러:** `400`(잘못된 파일/크기), `422`(검증 실패), `500`(추론 오류) — 본문에 `{ "error": { "code", "message" } }`

### 4.2 `GET /drugs/{category_id}`
- **응답 200:** 해당 `category_id`의 `drug_master` 카드(`(M)` 컬럼 묶음). 미존재 시 `404`.

### 4.3 `GET /health`
```json
{ "status": "ok", "model_version": "yolo11s_r1024_v3", "model_loaded": true }
```
- 워밍업·모니터링·배포 검증에 사용(§8).

### 4.4 `POST /feedback` (심화)
- **요청:** `{ "prediction_item_id", "is_correct", "corrected_category_id?", "comment?" }`
- **응답 201:** `{ "feedback_id", "status": "received" }` → `feedback` 테이블(문서 A)

---

## 5. 기술 스택

| 레이어 | T1 | T2 |
| --- | --- | --- |
| UI | **Gradio** (자동) | **Next.js** (Vercel) |
| API | (Gradio 내장) | **FastAPI** + Pydantic |
| 추론 | PyTorch·Ultralytics | PyTorch·Ultralytics (옵션 ONNX Runtime) |
| 후처리 | NMS / max_det=4 | 동일 |
| 저장 | 정적 JSON | **Supabase(Postgres)** |
| 패키징 | (Space repo) | **Docker** |
| 호스팅 | **HF Spaces** | Render·Railway·HF Docker(백) + Vercel(프론트) |
| CI/CD | git push 자동빌드 | **GitHub Actions** |

---

## 6. 배포 런북 (DevOps) — 데모 보험

### 6.1 T1 (HF Spaces)
| 단계 | 절차 |
| --- | --- |
| build | Space repo에 `app.py`·`requirements.txt`·`serving_bundle/` push → 자동 빌드 |
| verify | `/health` 또는 샘플 추론으로 정상 확인 |
| rollback | 직전 커밋으로 revert (Space는 git 기반) |
| secret | (T1은 비공개키 불필요) Space 설정의 변수만 |

### 6.2 T2 (Docker 백엔드 + Vercel 프론트)
| 단계 | 절차 |
| --- | --- |
| build | `docker build` → 레지스트리 push (또는 Render git 연동 자동) |
| deploy | main 머지 시 GitHub Actions → 백엔드 재배포, Vercel은 push 자동 |
| verify | `/health` 그린 + smoke test(샘플 이미지 1장) |
| rollback | 백=직전 이미지/커밋 재배포, 프론트=Vercel 대시보드 즉시 롤백 |
| env/secret | `MODEL_PATH`, `SUPABASE_URL/KEY`, `CORS_ORIGINS` — 커밋 금지, 환경변수 |

### 6.3 환경 변수
```
MODEL_VERSION=yolo11s_r1024_v3
MODEL_PATH=/app/serving_bundle/yolo11s_r1024_v3
SCORE_THRESHOLD=0.25
NMS_IOU=0.5
MAX_DET=4
SUPABASE_URL=...        # T2
SUPABASE_KEY=...        # T2 (서버 전용)
CORS_ORIGINS=https://<frontend-domain>
```

---

## 7. 성능 · 지연 예산 (NFR-03)

| 구간 | 목표(T1 CPU 기준) | 비고 |
| --- | --- | --- |
| 업로드·디코드 | ~50ms | 파일 크기 제한으로 통제 |
| 전처리 | ~30ms | resize·normalize |
| **추론** | ~300~1500ms | 콜드 스타트 시 별도(수 초) |
| 후처리·조인 | ~20ms | NMS·drug_master 조인 |
| **총 체감** | **목표 < 2~3s** | 워밍 상태 기준 |

- 지연 최적화 레버: 입력 해상도↓(정확도 트레이드오프), ONNX export(T2), GPU Space(필요 시 한시).
- **콜드 스타트가 데모 최대 적** → §8 워밍업으로 제거.

---

## 8. 데모 안정성 (라이브 실패 방지) ★

| 장치 | 내용 |
| --- | --- |
| 워밍업 | 발표 직전 `/health`·샘플 추론 1~2회로 Space/서비스 깨우기 |
| 헬스 게이트 | `/health` 그린 확인 후 데모 시작 |
| 프리로드 샘플 | 업로드 실패 대비 **내장 샘플 이미지 버튼** 제공(라이브 업로드 의존 제거) |
| 그레이스풀 에러 | 추론 실패 시 크래시 대신 명확한 에러 카드 |
| 폴백 영상 | 최악의 경우 **녹화 데모** 대체본 준비(상세는 문서 C 발표 스크립트) |

---

## 9. 관측 가능성 (심화)

- **로깅:** `prediction`(latency_ms·model_version·num_detections)·`prediction_item`(category_id·score) 비동기 적재.
- **지표:** 추론 지연 p50/p95, 검출 수 분포, score 분포, 에러율.
- **드리프트 신호:** score 분포·클래스별 빈도의 시간 변화 → 모델 노후 감지.
- 원본 이미지 미저장(`image_hash`만) — 프라이버시(문서 A §9 / 정책은 문서 C).

---

## 10. 피드백 · 재학습 루프 (심화) — FR-11

```
[/feedback] → feedback 테이블 → [큐레이션: is_correct=false 수집·검수]
   → [재학습 데이터셋 보강] → [학습 → 신모델] → model_registry 등록
   → [is_serving 스왑] → [재배포(§6)]  ↺
```
- 설계만으로도 포폴 깊이↑(구현은 여력 시). model_registry의 `is_serving` 플래그로 무중단 모델 교체 표현.

---

## 11. 보안 (인프라 관점)

| 항목 | 조치 |
| --- | --- |
| 입력 검증 | 허용 타입(png/jpg)·최대 크기(10MB)·디코드 실패 거부 |
| 레이트 리밋 | 공개 데모 남용 방지(간단 IP/세션 기반) |
| CORS | 프론트 도메인만 허용(`CORS_ORIGINS`) |
| 시크릿 | 환경변수 관리, 저장소 커밋 금지(서버 전용 키 분리) |
| 데이터 | 원본 이미지 미저장(해시만) |

---

## 12. SRS / PRD / A 추적

| 본 설계 | 충족 |
| --- | --- |
| §1 토폴로지 T1/T2 | 실행전략 §6.2 |
| §2 패키징 번들 | model_registry(A), NFR-01(재현성) |
| §3 추론 파이프라인 | 실행전략 §6.3, FR-07, 불변식(A §2) |
| §4 API 명세 | prediction/_item/drug_master(A), FR-03 |
| §7 지연 예산 | NFR-03 |
| §8 데모 안정성 | 실행전략 ⑥, (문서 C 연계) |
| §10 피드백 루프 | FR-11, NFR-07 |
| §11 보안 | (문서 C 프라이버시 연계) |

---

## 13. 다음 액션

1. **T1 `app.py` 스캐폴딩** — serving_bundle 로드 + `predict()` + Gradio UI (발표 보장선)
2. **FastAPI `/predict` 구현**(T2) — §4 스키마, 후처리·조인 파이프라인 §3
3. **`/health` + 워밍업 스크립트** — 데모 보험 §8
4. (이어서) **문서 C UX·데모 설계** — 화면·디스클레이머·발표 스크립트로 서비스면 완결

---

*모델은 `serving_bundle`로 패키징되고, `drug_master`를 조인해 약 정보를 반환하며, `/health`로 깨운 뒤 라이브로 시연된다. 여기서 모델이 서비스가 된다.*
