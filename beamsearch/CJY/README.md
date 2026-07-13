# beamsearch/CJY — 최중열 (Experimentation Lead(→서빙))

개인 탐색 공간. 구조·도구·방식 자유. `shared`(채점기·가드·SSOT)만 재사용한다.

- 실행 예: `from shared.eval.local_map import ...` · `python -m shared.ssot.build_drug_master ...`
- 공통 데이터: `team/data/raw`(원본)·`team/data/processed`(SSOT) 참조
- 승격 후보 노트북은 의사결정 md 정리해 `team/notebooks/`로.

---

# beamsearch/CJY 실험 정리 — 경구약제 Object Detection

작성: 최중열 (Experimentation Lead) · 2026-07-10 기준
공간: `beamsearch/CJY/` (개인 탐색 공간, `shared` 재사용 원칙)

이 문서는 CJY 공간에서 진행한 실험 전체를 시간순으로 정리한 것이다.
무엇을, 왜, 어떤 순서로 실험했고, 각 단계에서 무엇을 배웠는지, 그리고
팀원이 같은 실험을 재현하려면 어떻게 해야 하는지를 담는다.

---

## 0. 한눈에 보기

```
Phase 1  베이스라인      exp001~005   5개 모델 tiny 비교 → YOLO11n 채택
Phase 2  데이터 증강     exp010~011   Kaggle 단독 vs +AIHub → 성능 점프, 그러나 의심
Phase 3  누수 정리       exp016       감사(audit) + 유사중복 그룹화 → 정직한 검증셋
Phase 4  검증 강화       exp020 kfold 5-fold 교차검증으로 분산 확인
Phase 5  최적화          hpo, exp022  Optuna HPO → YOLO11n/11s best config
Phase 6  대안 모델       exp012~017   SSD/RetinaNet/FCOS/RF-DETR 증강 재실험
Phase 7  실도메인 도전   r0, r1 시리즈 실사용 사진 평가 → 합성 데이터 실험 (진행 중)
```

**현재 베스트 (Kaggle 검증 기준):** `exp016_yolo11s_aug_clean_hpo_best`
— val mAP@75:95 **0.98** (60ep, 누수 제거 데이터 + HPO 파라미터)

---

## 1. Phase 1 — 모델 베이스라인 비교 (exp001~005)

작은 서브셋(tiny)으로 5개 검출 모델을 같은 조건에서 비교해 본 실험 대상을 정했다.

| 실험 | 모델 | epochs | val 점수* | 비고 |
|---|---|---|---|---|
| exp001 | YOLO11n | 20 | mAP@50:95 0.957 | 최고, 빠름 |
| exp002 | SSD300 | 20 | mAP@50:95 0.898 | |
| exp003 | Faster R-CNN | 5 | **0 (발산)** | train_loss NaN |
| exp004 | RetinaNet | 60 | mAP@75:95 0.894 | |
| exp005 | FCOS | 60 | mAP@75:95 0.927 | |

*tiny 실험들은 서브셋·메트릭 기준이 이후 실험과 달라 상대 비교용으로만 볼 것.

**Faster R-CNN NaN 원인 (디버깅 기록):** 라벨 category_id 비연속 + degenerate
box(폭/높이 0) + albumentations 버전 차이가 겹친 gradient explosion. 이 과정에서
category_id 체계 문제를 발견한 게 Phase 2의 데이터 정비로 이어졌다.

**category_id 통일:** AIHub 변환기가 "발견 순서"로 id를 부여해 Kaggle의 K-코드
기준과 불일치 → `parse_kcode`로 파일명에서 K-코드를 직접 파싱해 통일. 제출
양식(56 클래스)도 이때 맞춤.

## 2. Phase 2 — 데이터 증강 실험 (exp010~011)

| 실험 | 데이터 | val mAP@75:95 |
|---|---|---|
| exp010 | Kaggle train 단독 | 0.842 |
| exp011 | Kaggle + AIHub 조합 1,3 | 0.989 |

15%p 점프가 나왔지만 **너무 좋아서 의심**했다. 두 가지 가능성: (1) AIHub에
대회 테스트셋과 같은 사진이 섞임(진짜 컨닝), (2) train/val 분할 누수로 검증
점수만 부풀려짐. → Phase 3.

## 3. Phase 3 — 누수 감사와 유사중복 그룹화 (exp016)

상세는 **`docs/data_leakage_prevention.md`** (파이프라인·사용법) 참고. 요약:

- **테스트셋 대비 감사** (`src/audit_leakage.py::audit`): 픽셀 SHA1 → 지각해시
  (dHash 256-bit, Hamming≤12) → 32×32 RMSE(≤0.06) 재검증의 3단계 대조.
  AIHub발 근접중복 **6장 발견 → 제거**. Kaggle train 자체의 내재적 중복은
  대회 공통 조건이므로 제거하지 않음. exp011은 "컨닝 아님" 판정.
- **유사중복 그룹화** (`src/data/subset.py::build_leakage_safe_groups`): 같은
  트레이에서 알약 1개만 바꿔 찍은 "조합 코드는 다른데 사진은 같은" 쌍이
  val의 ~30%였음 → 지각해시+union-find로 그룹을 병합해 그룹 단위 분할.
  분할 후 근접중복 0% 재검증 셀 포함.

| 실험 | 조건 | val mAP@75:95 |
|---|---|---|
| exp011 | 그룹화 前 | 0.9895 (부풀려짐) |
| exp016 | 그룹화 後 (clean) | 0.9845 (정직) |

점수가 내려간 것이 목적이다 — 이후 모든 비교는 이 clean 분할 위에서 수행.

## 4. Phase 4 — K-Fold 교차검증 (exp020)

clean 그룹 기준 5-fold (`create_group_kfold`). fold별 mAP@75:95 =
0.9685 / 0.9654 / 0.9719 / 0.9687 / 0.9602 (**평균 0.967, 표준편차 ~0.004**).
분산이 작아 단일 홀드아웃 검증으로도 신뢰 가능하다고 판단, 이후 실험은
비용을 아끼기 위해 홀드아웃으로 진행.

## 5. Phase 5 — 하이퍼파라미터 탐색 (hpo, exp016_hpo_best, exp022)

`src/hpo.py` (Optuna 베이지안 탐색) 사용. 핵심 설계:

- 탐색 단계는 **짧은 epoch(YOLO 8ep)으로 "방향"만** 잡고, best 조합으로 전체
  epoch config를 새로 만들어 재학습 (`materialize_best_config`). 짧은 학습
  점수를 최종 성능으로 믿지 않는다.
- trial은 `experiments/hpo/<exp>_trial{N}/`에 격리 저장 → 기존 metrics.json
  비교 코드 재사용.
- 탐색 공간: lr, weight_decay, batch_size, optimizer (`DEFAULT_SEARCH_SPACES`).

| 대상 | trials | 결과 |
|---|---|---|
| YOLO11n (exp016 기반) | 15 | best 재학습 mAP@75:95 **0.9916** (60ep) |
| RetinaNet (exp014 기반) | 10 | hpo_best config 생성 |
| YOLO11s (exp022 기반, AIHub 1~6) | 15 | best trial 0.9928 → 재학습 **0.9944** |

YOLO11s 최종 best 파라미터: AdamW, lr≈1.7e-4, wd≈3.9e-5, batch 32
(`configs/experiment/exp016_yolo11s_aug_clean_hpo_best.yaml`).

## 6. Phase 6 — 대안 모델 증강 재실험 (exp012~015, 017)

| 실험 | 모델 | epochs | val mAP@75:95 | 비고 |
|---|---|---|---|---|
| exp012 | SSD300 | 30 | 0.942 | |
| exp013 | Faster R-CNN | 30 | (기록 없음) | 재발산 → 재실험 필요 |
| exp014 | RetinaNet | 30 | 0.976 | |
| exp015 | FCOS | 30 | 0.984 | |
| exp017 | RF-DETR | 30 | 0.960 | epoch당 ~16분(느림) |

⚠️ **주의:** exp012~015는 그룹화 적용 전(_aug) 분할로 학습돼 val mAP@75가 1.0이
찍히는 등 **점수가 부풀려져 있다.** exp016 계열과 직접 비교하지 말 것.
공정 비교하려면 clean 분할로 재학습해야 한다.

**RF-DETR MPS 이슈:** deformable attention의 `grid_sampler_2d_backward`가
PyTorch MPS에 미구현 → `PYTORCH_ENABLE_MPS_FALLBACK=1`을 torch import 전에
설정해 해당 연산만 CPU 폴백 (`src/train_hf.py` 상단 + 노트북 첫 셀).

## 7. Phase 7 — 실도메인(R0) 평가와 합성 데이터 (r1 시리즈, 진행 중)

검증 0.99가 나와도 **실사용 사진(폰 촬영, 실배경, 알약 겹침)** 에서 통하는지는
별개다. 직접 촬영한 평가셋(R0, 78장/392박스, class-agnostic)으로 측정:

| 단계 | 학습 데이터 | R0 recall(IoU≥0.5) |
|---|---|---|
| baseline (exp016 계열) | 스튜디오 데이터만 | **0.077** |
| r1_pill_agnostic | + C3PI 실사진(SAM 자동라벨 3,004장) | 0.107 |
| r1b_kor | + 한국 약 SAM 컷아웃 겹침 합성(절차적 배경) | **0.181** |
| r1c (kor_on_real) | 합성 배경을 실배경(C3PI)으로 교체 | 0.158 |
| r1d_merged | 세 합성셋 합침 (8,058장) | 진행 중 |

관련 모듈: `src/synth/` (sam_label, cutouts, kor_synth, compose, blend,
randomize 등), 노트북 `r0_eval.ipynb`, `r1_finetune.ipynb`, `r1b_kor_synth.ipynb`.

**중간 결론:** 스튜디오→실사용 도메인 갭이 진짜 병목. 합성만으로는 recall
~0.18이 천장으로 보이며, r1d로 0.18을 못 넘으면 **실제 한국 약 학습 사진
200~500장 수집·라벨**이 다음 단계다.

---

## 8. 재현 방법

### 환경
- 루트 `pyproject.toml` 기준 `uv sync` (optuna 포함). Mac은 학습 시
  `device='mps'` 명시 필수 — 안 하면 CPU로 떨어져 수십 배 느려진다.
- Kaggle 데이터: `~/.cache/kagglehub/competitions/ai12-level1-project/...`
  (kagglehub로 자동 다운로드). AIHub 원본은 개별 보유 필요.

### 메인 실험 재현 (exp016 계열)
1. `beamsearch/CJY/notebooks/run_experiment_augmented.ipynb`를 위에서부터 실행.
   1~4단계(변환→감사→그룹화→분할·병합)가 누수 안전 데이터(`train_aug`/`val_aug`)를
   만들고, 검증 셀이 근접중복 0%를 확인한다. **이 셀들은 수정하지 말고 그대로
   쓸 것** — 직접 분할하면 누수가 재발한다.
2. 이후 학습 셀에서 `configs/experiment/`의 원하는 yaml을 골라 실행.
   결과는 `experiments/<exp_name>/`에 config.yaml + metrics.json으로 저장된다.

### HPO 재현
`run_hpo.ipynb`(11n) / `run_hpo_full_hub.ipynb`(11s, AIHub 1~6). 데이터 준비
셀은 위와 동일하고, `run_hpo(base_config=..., n_trials=15, hpo_epochs=8)` →
`materialize_best_config()` → best config로 재학습 순서.

### 제출 파일 생성
`generate_submission_yolo_hpo.ipynb` 또는 각 실험 노트북의 8단계 셀:
`restrict_class_map`(56클래스 제한) → `run_inference` → `save_submission`.
제출 전 검증 셀에서 이미지 수/카테고리 범위를 확인한다.

### 클린 데이터셋만 받아서 쓰기 (팀원용)
직접 파이프라인을 돌리지 않아도 `export_clean_dataset.ipynb`가 만든 zip
(`data/export_shared_clean/`)을 받아 `load_portable_coco()`로 바로 쓸 수 있다.
YOLO는 로컬에서 `prepare_yolo_dataset(..., symlink=False)` 한 번만 실행.

### 실도메인 평가 재현
`r0_eval.ipynb`에서 `WEIGHTS`만 평가할 모델의 best.pt로 바꿔 실행.
R0 평가셋 프로토콜은 `docs/realdomain_evalset_protocol.md` 참고.

---

## 9. 통찰 (이 프로젝트에서 배운 것)

**너무 좋은 점수는 버그처럼 다뤄라.** exp011의 +15%p는 절반이 거품(분할 누수)
이었다. 이상치 성능이 나오면 축하 전에 감사(audit)부터 — 그리고 감사는 일회성
확인이 아니라 파이프라인에 박아 매번 자동으로 돌게 해야 한다.

**분할 단위는 파일이 아니라 장면.** 같은 대상을 반복 촬영한 데이터는 파일명
기준 랜덤 분할이 반드시 누수를 만든다. 내용 기반(지각해시) 그룹화가 답이었다.

**검증 점수와 실사용 성능은 별개의 축.** clean 검증 0.99를 만들고도 실사용
사진 recall은 0.08이었다. 검증셋이 배포 환경을 대표하지 않으면 그 0.99는
"스튜디오 시험 만점"일 뿐이다. 실도메인 평가셋(R0)을 일찍 만들었어야 했다.

**합성 데이터는 갭을 좁히지만 공짜가 아니다.** 겹침·배경은 합성으로 만들 수
있었지만(0.077→0.181), 스튜디오 컷아웃의 "외형 갭"은 합성으로 안 지워졌다.
흥미롭게도 실배경(r1c)보다 절차적 다양 배경(r1b)이 더 나았다 — 배경의
"진짜스러움"보다 **다양성**이 모델이 배경에 의존하지 않게 만든다.

**HPO는 짧게 탐색, 길게 확정.** 8ep 탐색 점수와 60ep 최종 점수는 다르다.
탐색은 순위 비교용으로만 쓰고 best는 반드시 전체 epoch으로 재학습했다.

**하드웨어 함정 기록.** Mac에서 device 미지정 → CPU 학습(57시간 코스),
통합 메모리 초과 → 스와핑으로 CPU보다 느려짐(imgsz·batch 축소로 해결),
MPS 미구현 연산 → FALLBACK 환경변수. 학습이 "이상하게 느리면" 로그 상단의
device와 GPU_mem부터 볼 것.

---

## 10. 파일 맵

| 무엇 | 어디 |
|---|---|
| 실험 config | `beamsearch/CJY/configs/experiment/*.yaml` |
| 실험 결과 (config+metrics) | `beamsearch/CJY/experiments/<exp_name>/` |
| 누수 감사 | `beamsearch/CJY/src/audit_leakage.py` |
| 그룹화·분할·export | `beamsearch/CJY/src/data/subset.py`, `merge.py` |
| HPO | `beamsearch/CJY/src/hpo.py` |
| 합성 데이터 | `beamsearch/CJY/src/synth/` |
| 메인 노트북 | `notebooks/run_experiment_augmented.ipynb` |
| HPO 노트북 | `notebooks/run_hpo.ipynb`, `run_hpo_full_hub.ipynb` |
| K-Fold 노트북 | `notebooks/run_experiment_kfold.ipynb` |
| 실도메인 평가 | `notebooks/r0_eval.ipynb` |
| 제출 생성 | `notebooks/generate_submission_yolo_hpo.ipynb` |
| 제출 이력 | `beamsearch/CJY/submissions/*.csv` |
| 누수 방지 상세 문서 | `beamsearch/CJY/docs/data_leakage_prevention.md`* |
