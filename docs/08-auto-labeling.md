# 오토라벨링 실험 (E9) — 문서 08

> **코드잇 스프린트 AI 엔지니어링 12기 · 초급 프로젝트 · 3팀**
> 목적: 라벨 없는 공개 의약품 이미지를 **오토라벨링 파이프라인**으로 학습 가능한 형태로 만든다.
> 성격: 점수 기여보다 **데이터 엔지니어링 실무 역량**(라벨링 파이프라인·파운데이션 모델·HITL·self-training) 축적. 발표/보고서 소재.
> 조사·설계: 이형기 · 데이터 통제: 홍우석(DE) · 작성일 2026-06-29

---

## 0. 왜 하는가 (서사 연결)

우리 서사 — *"리더보드 점수가 아니라 환자가 쓸 수 있는 서비스를 만들었다"* — 에 **데이터 엔지니어링 한 겹**을 더한다: *"라벨 없는 식약처 공개 이미지를, 우리가 만든 오토라벨링 파이프라인으로 학습 데이터로 전환했다."* 소량이라도 다음을 직접 경험한다.
- 라벨링 파이프라인 설계(클래식 CV ↔ 파운데이션 모델)
- 메타데이터 조인으로 **클래스 자동 부여**(세미 오토라벨링)
- HITL(사람 검수) 워크플로우
- pseudo-labeling 기반 **self-training** 루프

---

## 1. 실험 설계

| 트랙 | 방법 | 위치 | 역할 |
| --- | --- | --- | --- |
| **①** | 클래식 CV (Otsu+contour) | `team/src/data/autolabel_cv.py` | **메인** — 흰 배경 단일 알약 bbox 자동 생성 |
| **②** | 파운데이션 모델 (Grounding DINO+SAM) | `src/data/autolabel_foundation.py` | **비교군** — 성능·속도 대조 |
| **③** | pseudo-labeling (우리 모델) | `src/data/autolabel_pseudo.py` | **실습** — self-training 소량 체험 |
| 평가 | IoU vs 정답 | `src/data/autolabel_eval.py` | ①/② 품질 수치화 |

**핵심 통찰 — "세미 오토라벨링":** 낱알식별 데이터는 **클래스가 메타데이터로 이미 주어져** 있다. 따라서 우리가 자동화할 것은 **bbox 생성뿐**이다. → 난이도가 낮아 파이프라인 입문에 최적.

**데이터 출처(docs/07 §3 연계):**
- 이미지: 식약처/약학정보원 **낱알식별** 단일 알약(흰 배경) → `data/raw/nedrug_single/`
- 메타: 「의약품 낱알식별 정보」(품목기준코드·품목명·모양·색 등)
- 정답(평가용): AI Hub 경구약제 **단일**(bbox 보유) → ①/②의 bbox 품질 측정 기준

**클래스 매핑 주의(중요):** 낱알식별은 **품목기준코드(ITEM_SEQ)**, 우리 라벨은 **dl_idx 기반 category_id**로 식별자가 다르다. 둘을 잇는 **크로스워크**(`{품목기준코드: category_id}`)가 있어야 우리 40클래스에 직접 합류 가능. → **`shared/ssot/build_crosswalk.py`** 가 `drug_master` 기반으로 생성한다(직접→EDI→제품명 3단 폴백). 크로스워크가 없으면 해당 데이터는 **미매핑(연습·일반화용)** 으로 처리(스크립트가 category_id=-1로 표시).

---

## 2. 파이프라인

```
낱알 이미지 ─┐
            ├─[① CV: Otsu+morph+contour]─→ bbox ─┐
            └─[② 파운데이션: prompt "pill"]─→ bbox ┘
                                                   ├─ 메타 조인(품목기준코드 → 약정보 + category_id)
                                                   ├─ COCO 라벨 출력
                                                   ├─ (선택) QC 시각화 → 사람 검수(HITL)
                                                   └─ 평가(IoU vs AI Hub 단일 정답)
[③ pseudo] 라벨없는 이미지 → 우리 모델 예측 → conf 임계 → pseudo COCO → 재학습
```

모든 트랙은 **금지경로 가드**(조합/TL_2/TS_2 자동 차단)와 **동일한 COCO 출력 계약**을 공유 → 비교가 공정.

---

## 3. 평가지표 (발표/보고서용)

| 지표 | 의미 | 도구 |
| --- | --- | --- |
| **mean IoU** | 자동 bbox vs 정답 정확도 | `autolabel_eval.py` |
| **match@0.75** | 우리 대회 임계(0.75)와 동일선상 통과율 | `autolabel_eval.py` |
| **검출 수율** | bbox 생성 성공/전체 | 각 스크립트 리포트 |
| **속도(ms/img)** | 처리 효율 | 각 스크립트 Timer |
| **수작업 절감** | 자동→검수로 줄인 라벨링 시간(정성) | HITL 기록 |

**비교표 (채워서 보고서에)**

| 방법 | mean IoU | match@0.75 | 수율 | ms/img | 비고 |
| --- | --- | --- | --- | --- | --- |
| ① 클래식 CV | | | | | 모델 불필요, 흰배경 강함 |
| ② 파운데이션 | | | | | 일반 배경 강함, 무거움 |

> 예상 가설: 흰 배경 단일 알약에선 **① CV가 빠르고 충분히 정확**, 배경이 복잡해질수록 **② 파운데이션이 우위**. 이 트레이드오프 자체가 좋은 발표 포인트.

---

## 4. 실행 방법

```bash
# ⓪ 크로스워크 생성 (품목기준코드 → category_id) — 오토라벨 결과를 40클래스로 합류
python -m shared.ssot.build_crosswalk \
    --drug-master data/processed/drug_master.json --nedrug-meta data/raw/nedrug_meta.csv \
    --out data/processed/itemseq_to_category.json

# ① 클래식 CV (메인)
python -m team.src.data.autolabel_cv \
    --images data/raw/nedrug_single --metadata data/raw/nedrug_meta.csv \
    --crosswalk data/processed/itemseq_to_category.json \
    --out data/processed/autolabel_cv.json --viz-dir outputs/autolabel_cv_viz

# ② 파운데이션 (비교군 — 별도 설치: pip install autodistill autodistill-grounded-sam)
python -m team.src.data.autolabel_foundation \
    --images data/raw/nedrug_single --prompt pill \
    --out data/processed/autolabel_foundation.json

# ③ pseudo-labeling (실습 — 우리 모델 가중치 필요)
python -m team.src.data.autolabel_pseudo \
    --weights outputs/e1_best.pt --images data/raw/unlabeled \
    --out data/processed/pseudo_labels.json --conf 0.5 --max-det 4

# 품질 평가 (①/② 각각 vs 정답)
python -m team.src.data.autolabel_eval \
    --pred data/processed/autolabel_cv.json --gt data/processed/aihub_single_filtered.json
```

---

## 5. QC · 검수(HITL) 규약

- 자동 bbox는 그림자·반사·테두리로 오차 발생 → `--viz-dir` 결과를 **사람이 표본 검수**.
- pseudo-label은 **보수적 conf 임계**(정밀도 우선)로 채택, 채택분만 학습 합류.
- 검수 표본·수정 건수·소요 시간을 `experiments/decision-log.md`에 기록(수작업 절감 근거).

---

## 6. 리스크 · 주의

- **단일 → 멀티 갭**: 낱알은 1알/이미지 → 우리 멀티(최대 4알) 분포와 다름 → **구조적 Copy-Paste 증강**(`team/src/data/augmentor.py`)으로 보완.
- **라이선스·출처**: 공공데이터 활용 시 출처 명시. 상업/재배포 조건 확인.
- **미매핑 클래스**: 크로스워크 없으면 우리 40클래스에 직접 기여 못함(연습·일반화용으로만).
- **금지 데이터 무관**: 낱알식별은 AI Hub 조합 세트와 별개. 그래도 코드 가드는 항상 작동.

---

## 7. 산출물 · 포트폴리오 포인트

- 오토라벨링 파이프라인 4종 스크립트(`src/data/autolabel_*.py`) + COCO 출력
- ① vs ② **정량 비교표**(IoU·속도) → 발표 슬라이드 1장
- 메타 조인으로 만든 **약정보 포함 라벨**(서비스 인텔리전스와 연결, docs/04·06)
- "라벨 없는 공개 데이터 → 학습 데이터" 전환 스토리

---

## 8. 다음 액션

1. (홍우석) 낱알식별 이미지·메타 확보 → `data/raw/nedrug_single/`, `data/raw/nedrug_meta.csv`
2. 크로스워크 생성: `python -m shared.ssot.build_crosswalk`(drug_master의 품목기준코드 ↔ category_id, 직접→EDI→제품명 폴백)
3. ① 실행 → QC 표본 검수 → ② 비교 → `autolabel_eval`로 표 채우기
4. E1 모델 확보 후 ③ pseudo 소량 실습 → decision-log 기록
