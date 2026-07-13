# 헬스잇(Health Eat) 경구약제 객체 검출 — 팀 모노레포

> **코드잇 스프린트 AI 엔지니어링 12기 · 초급 프로젝트 · 3팀**
> **ULTRA CAPSHYONG ITEM WITH 4 VALUES**

사진 속 **최대 4개 알약의 클래스 + 바운딩 박스**를 검출하고, 검출 결과를 실제 약 정보로 변환하는 **웹 서비스**까지 만드는 프로젝트.

- **과제:** Object Detection (이미지당 0~4개, COCO 포맷) · **지표:** mAP@[0.75:0.95] (Kaggle)
- **기간:** 2026-06-26 ~ 07-13 · 중간발표 07-07 · 최종발표 07-14

---
주요 보고서 및 협업일지 링크
https://app.notion.com/p/1_3-38bb13f77dbd80789762f37a573d2c1e?source=copy_link

## ★ 레포 구조 (정본 — 구조도는 이 문서 1곳에만)

> 다른 문서는 구조를 **다시 그리지 않고** 이 절을 링크로 참조한다(드리프트 방지). 코드 실제 위치·실행 명령의 정본은 아래 **[코드 위치 맵](#-코드-위치-맵)**.

```
repo/
├── README.md              # ★ 구조 SSOT + 팀 자산 허브 (이 문서)
├── .gitignore  requirements.txt  pyproject.toml  Makefile
│
├── docs/                  # 팀 설계 문서 00~08 (루트 공유 — 개인 폴더에 복제 X)
│   └── README.md
│
├── shared/                # 경기 규칙(얇게, 전원 공용) ★공정성 코어
│   ├── env.py             #   디바이스 자동 감지·시드
│   ├── guards/            #   banned.py — 금지 데이터 가드(조합/TL_2/TS_2) 정본
│   ├── ssot/              #   build_drug_master · build_crosswalk (category 계약)
│   ├── eval/              #   local_map.py — mAP@[0.75:0.95] 공통 심판
│   └── submit/            #   make_submission.py — 제출 포맷 검증(IR-06)
│
├── beamsearch/            # 탐색 — 각자 세계 (충돌 제로, 작업용 노트북)
│   ├── LTM/ LHK/ CJY/ JHB/ HWS/   # 개인 영역: 구조·도구·방식 자유
│   └── README.md
│
└── team/                  # 마일스톤 승격 결과 (전원 심사 통과분만) + 공통 데이터
    ├── notebooks/         #   ★승격된 서사 노트북(코드+출력+의사결정 md) = 제출물
    ├── data/              #   raw(미커밋)/processed(SSOT json)
    ├── src/               #   data · models · inference (승격 파이프라인)
    ├── service/           #   웹 서비스(app.py, serving_bundle)
    ├── configs/  experiments/  outputs/  report/
```

**3영역 원칙:** `shared`(처음부터 전원 공용 규칙) ↔ `beamsearch/<이니셜>`(개인 자유 탐색) ↔ `team`(승격물 + 공통 데이터). 공정성은 **같은 채점기·같은 금지 가드**를 `shared`로 공유해서 확보하고, 방식·구조·도구는 개인 영역에서 자유. 승격 워크플로우는 [docs/03 §2.4](./docs/03-execution-strategy.md).

---

## 🗺 코드 위치 맵

| 기능 | 모듈(실행) | 파일 |
| --- | --- | --- |
| 디바이스/시드 | `shared.env` | `shared/env.py` |
| 금지 가드(정본) | `shared.guards.banned` | `shared/guards/banned.py` |
| SSOT 생성 | `shared.ssot.build_drug_master` | `shared/ssot/build_drug_master.py` |
| 크로스워크 | `shared.ssot.build_crosswalk` | `shared/ssot/build_crosswalk.py` |
| 로컬 mAP | `shared.eval.local_map` | `shared/eval/local_map.py` |
| 제출 CSV·검증 | `shared.submit.make_submission` | `shared/submit/make_submission.py` |
| COCO→YOLO | `team.src.data.coco_to_yolo` | `team/src/data/coco_to_yolo.py` |
| 구조적 증강(단일→멀티) | `team.src.data.augmentor` | `team/src/data/augmentor.py` |
| 외부 단일 필터 | `team.src.data.filter_aihub` | `team/src/data/filter_aihub.py` |
| 오토라벨링 E9 | `team.src.data.autolabel_*` | `team/src/data/autolabel_*.py` |
| 학습 | `team.src.models.train` | `team/src/models/train.py` |
| 추론 | `team.src.inference.predict` | `team/src/inference/predict.py` |
| 웹 서비스 | — | `team/service/app.py` |

> 설계 문서(docs/)의 `src/…` 표기는 **논리적 파이프라인 명칭**이며, 실제 위치·명령은 이 표가 정본.

---

## 💻 개발·학습 환경 & 워크플로우

- **학습(공식): Runpod** — PyTorch 2.8 + CUDA 12.8 프리빌트(JupyterLab). GPU는 모델 규모에 맞춰 팀 상의.
- **로컬 개발:** VSCode + JupyterLab 등으로 학습 외 작업. Colab 유료 GPU 등 병용 가능.
- **디바이스 이식성:** `shared.env`로 자동 감지(CUDA→MPS→CPU). 개인 머신 사양은 규정하지 않음.
- **워크플로우:** 로컬에서 데이터·전처리·스모크까지 → push → **Runpod에서 GPU 학습** → 결과 push → Runpod 종료. (학습 외는 로컬, GPU 학습만 Runpod)

---

## ⚙️ 재현 절차 (개요)

```bash
pip install -r requirements.txt   # 또는  pip install -e .   (레포 루트에서 실행)

make ssot        # shared.ssot.build_drug_master → team/data/processed/{class_map,drug_master}.json
make prep        # team.src.data.coco_to_yolo → YOLO 학습 포맷 + data.yaml
make train       # team.src.models.train (configs/e1_baseline.yaml, device 자동)
make predict     # team.src.inference.predict → team/outputs/detections.json
make submit      # shared.submit.make_submission → team/outputs/submission.csv (검증)
make eval        # shared.eval.local_map (mAP@[0.75:0.95])
```

> 데이터 보강: `make crosswalk`·`make augment`·`make autolabel` (docs/07·08). 웹 데모: `make serve`. 실제 인자는 `Makefile` 참조.

---

## 📦 데이터 · 증강 · 오토라벨링

- **대회 데이터** → `team/data/raw/`(미커밋). **SSOT**(class_map/drug_master) → `team/data/processed/`.
- **외부 보강**: AI Hub 경구약제 **단일**(docs/07) · **구조적 Copy-Paste 증강**으로 단일→멀티 합성(`augmentor.py`).
- **오토라벨링(E9)**: 식약처/약학정보원 낱알식별(docs/08). MFDS 낱알 API 다운로더는 **본 레포에 넣지 않고 별도 진행**, 결과물만 `team/data/raw/`로 반입.
- ⛔ **금지 데이터셋**: `TL_2_조합.zip`/`TS_2_조합.zip` — `shared/guards/banned.py`가 코드 전반에서 자동 차단.

---

## 🤝 협업 방식 (사람 중심)

- **탐색:** 각자 `beamsearch/<이니셜>/`에서 자유롭게(구조·도구·방식). 공정성은 `shared` 채점기·가드로 보장.
- **승격:** 마일스톤마다 전원이 결과·과정을 비교·심사 → 우수안만 `team/`으로 승격. (게이트: docs/03 §2.4)
- **데일리 브리핑:** 매일 10:00~10:20. **Kaggle:** 팀 단위 제출, 1일 10회.
- **원격·승격:** 팀장 **이태민** 주도, 전원 평가로 결정(자동 머지 아님).

## ⚖️ 라이선스 · 금지

- 원본: AI Hub 경구약제 이미지 데이터(코드잇 가공 제공). 활용 시 출처 명시.
- AI Hub **조합 validation**은 오토라벨 **검증용으로만** 사용(주강사 확인, docs/07).
- 복제 금지(타 팀 주제·코드·결론) · 커밋 금지(원본 데이터·가중치·시크릿·개인 도구 설정).

---

## 👥 팀

| 이름 | 이니셜 | 역할 |
| --- | --- | --- |
| 이태민 | LTM | Project Manager (팀장) |
| 이형기 | LHK | Model Architect *(호칭 KAI · 과거 문서 JUSTIN 동일인)* |
| 장한빈 | JHB | Experimentation Lead |
| 최중열 | CJY | Experimentation Lead (→ 서빙/배포) |
| 홍우석 | HWS | Data Engineer |
