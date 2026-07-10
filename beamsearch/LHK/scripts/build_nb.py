import json
from pathlib import Path


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s}


def code(s):
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": s,
    }


cells = []

cells.append(
    md("""# 💊 HealthEat 경구약제 검출 — EDA ①: 도메인 앵커(대회 기본 데이터)

> **코드잇 AI 부트캠프 · 초급 프로젝트 3팀 · Model Architect(이형기/LHK)**
> Object Detection · 이미지당 0~4알 · 지표 **mAP@[0.75:0.95]** · GitHub freeze **2026-07-13 19:00**

**이 노트북의 범위:** 출처별 3분할 EDA 중 **(1) 대회 기본 데이터** — 유일하게 **테스트셋과 동일 출처·프로토콜**인 *도메인 앵커*이자 *유일한 val/test 소스*. (2) AI Hub 단일 / (3) 식약처 낱알은 별도 노트북.

*진행 원칙:* 섹션마다 **코드 → 결론(마크다운)**. 무거운 실행·가설검증·피벗 지점은 사람 확인 후 다음 단계.""")
)

cells.append(
    md("""## 🎯 목표 & 성공지표

- **1차:** 유효 제출 + Kaggle 점수(floor) — 파이프라인 관통(E1).
- **2차(중간발표 7/7):** 로컬 val mAP@[0.75:0.95] 베이스라인 대비 유의 향상.
- **최종(7/13):** 우상향 곡선 + **각 향상의 원인 설명**(순위 자체보다 원인 추적이 평가 핵심).

절대 수치 목표는 베이스라인 제출 후 확정(고-IoU 특성).""")
)

cells.append(
    md("""## 🧭 전략 (요약)

- **빔서치 협업 루프:** 전원 병렬 탐색 → 동일 채점기로 수렴(beam 1~3) → 발전/제외/주입.
- **3-출처 데이터 전략:** **(1) 도메인 앵커**(=본 노트북, 유일 val/test 소스) · **(2) AI Hub 단일**(외형 다양성 + 단일→멀티 Copy-Paste *합성 소스*, 애블레이션) · **(3) 식약처 낱알**(오토라벨 → 서비스 커버리지, 층화평가·시간박스).
- **이중트랙 모델:** 대회=앙상블(WBF) · 서빙=단일 경량. 후보 근거는 docs/02 §5.2 참조(YOLO26/YOLO11/Cascade R-CNN/DINO·Co-DETR/RT-DETR/DINOv2).""")
)

cells.append(
    md("""## 🗺 실행계획 (본 노트북)

`§0 환경·재현성` → `§1 데이터 인벤토리·정합성` → `(b) 클래스 분포·조합 공기출현` → `(c) bbox 크기/위치/개수` → `(e) 배경·조명 메타(도메인갭 예비)` → `종합·가설 갱신`.

> 채택/제출의 토대인 **SSOT(class_map·drug_master)** 는 이미 생성됨(train-56, `category_id = int(dl_idx)`).""")
)

cells.append(
    md("""## 💡 초기 가설 (EDA로 검증/기각)

- **H1 저데이터:** train 232장 → 과적합 위험, 전이학습·증강 필수.
- **H2 고-IoU:** mAP@[0.75:0.95] → **localization 정밀도가 지표 지배**.
- **H3 데이터 천장·불균형:** 특정 클래스 과다 → 롱테일, 천장 클래스 존재 예상.
- **H4 색 불안정:** 촬영 WB로 색상 단서 불안정 → 형태·각인 우선.
- **H5 (1)↔(2) 도메인갭:** 앵커는 멀티·실촬영, AI Hub 단일은 스튜디오 → 갭 정량화 필요.
- **H6 조합 편향:** 조합 이미지의 약품 공기출현이 무작위가 아닐 수 있음(리더보드 일반화 리스크).""")
)

cells.append(
    md("""## 📌 착수 전 사전 발견 요약

### (1) 도메인 앵커 — 정합성·구조 (직접 집계 확인)
- **train_images 232장 = 전부 조합(멀티) 이미지**(파일명 K-코드수 3:157 / 4:75). **테스트 도메인과 동일 = 베이스라인은 합성 없이 바로 학습 가능.**
- annotations = **pill 단위 763 JSON**(각 1 image + 1 annotation), 114 그룹. **누락 0 · 무효 bbox 0 · 전부 976×1280.**
- 이미지당 객체수 `{2:7, 3:151, 4:74}` (max 4). **파일명 약품수 > 실제 bbox수인 ~7장 = 품질 플래그**(오클루전/미어노테이션 의심).
- **`category_id = int(dl_idx)`(실제 약품코드), train distinct 56종.** Test=40 이지만 **로컬 도출 불가(test 무라벨)** → **train-56을 실전 작업공간**으로 확정. 채점기가 40에 대해서만 점수(40에 없는 예측은 오탐일 뿐 제출 유효). **팀원 제출 CSV 2건 모두 56⊆56** 로 이를 실증(40 ⊆ 56 해석 지지).
- **심한 불균형:** `3351`(일양하이트린정)이 763객체의 **약 20%** → 롱테일.
- **메타데이터 금광:** dl_name/material/di_class_no/otc/shape/color/**print_front(각인)** + **back_color/light_color/drug_dir/camera** → drug_master·서비스카드·EDA 층화 근거.

### (2) AI Hub 단일 80장(8세트) 육안 리뷰 — 예비
- **배경 = 의도적 변주 축**(같은 약을 여러 배경색; 암↔명·다색 전 범위) → 배경 강건성엔 유리, 분할은 고정배경 가정 불가.
- **WB가 샷마다 흔들려 알약 겉색 불안정**(같은 약이 tan↔white↔노랑) → 색 비의존, 형태·각인 우선(H4 실증 예비).
- **촬영 리그 2종**(평면 무광 / 원형 디퓨저+비네팅), **반투명 연질캡슐**은 CV 분할 실패 → **오토라벨 CV+SAM 하이브리드 필수**(docs/08 ②파운데이션 정규화).
- 노출·스케일·각인(음각/라틴·한글 인쇄) 다양. → (2)→(1) 브리징은 **현실배경 Copy-Paste**가 담당.""")
)

cells.append(md("## 🔧 0. 환경 · 재현성 · device"))

cells.append(
    code("""import os, sys, json, warnings
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

warnings.filterwarnings("ignore")

# 한글 폰트: macOS 내장 AppleGothic (설치 불필요) + 마이너스 깨짐 방지
mpl.rcParams["font.family"] = "AppleGothic"
mpl.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid", font="AppleGothic")

# 재현성
SEED = 42
np.random.seed(SEED)

# device (MPS 우선; EDA 단계라 torch 미설치여도 무관)
try:
    import torch
    DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
except Exception:
    DEVICE = "cpu (torch 미설치 — EDA 단계)"

# 경로
BASE = Path("/Users/macbook/dev/learning/codeit/01_Proj_HealthEat_cla")
DATA_ROOT = BASE / "01_data/01_sprint_ai_project1_data"
SSOT = BASE / "project1-3team/beamsearch/LHK/data/processed"
TRAIN_IMG, TRAIN_ANN, TEST_IMG = DATA_ROOT/"train_images", DATA_ROOT/"train_annotations", DATA_ROOT/"test_images"

print("pandas", pd.__version__, "| numpy", np.__version__, "| cv2", cv2.__version__, "| mpl", mpl.__version__)
print("device:", DEVICE, "| seed:", SEED, "| font:", mpl.rcParams["font.family"])
print("data:", DATA_ROOT.exists(), "| ssot:", (SSOT/"drug_master.json").exists())""")
)

cells.append(
    md(
        "## 📦 1. 데이터 인벤토리 & 정합성 (자동 재확인)\n\n디스크 실측·COCO 파싱·불변식을 코드로 재확인한다(사전 발견을 자동 검증)."
    )
)

cells.append(
    code("""# 파일 수
train_imgs = sorted(p.name for p in TRAIN_IMG.glob("*.png"))
test_imgs  = sorted(p.name for p in TEST_IMG.glob("*.png"))
ann_files  = list(TRAIN_ANN.rglob("*.json"))
print(f"train_images={len(train_imgs)} | test_images={len(test_imgs)} | annotation JSON={len(ann_files)}")

# 어노테이션 → 객체 단위 DataFrame
rows = []
for jf in ann_files:
    d = json.loads(jf.read_text(encoding="utf-8"))
    im = d["images"][0]
    for a in d["annotations"]:
        rows.append(dict(file_name=im["file_name"], width=im["width"], height=im["height"],
                         category_id=int(a["category_id"]), bbox=a["bbox"],
                         back_color=im.get("back_color"), light_color=im.get("light_color"),
                         drug_dir=im.get("drug_dir")))
ann = pd.DataFrame(rows)
print("객체수:", len(ann), "| 고유 이미지:", ann.file_name.nunique(), "| 고유 클래스:", ann.category_id.nunique())

# 이미지당 객체 수
per_img = ann.groupby("file_name").size()
print("이미지당 객체수 분포:", dict(per_img.value_counts().sort_index()))

# bbox 유효성 · 해상도
bad = ann.bbox.apply(lambda b: not (isinstance(b, list) and len(b)==4 and b[2]>0 and b[3]>0)).sum()
print("무효 bbox:", int(bad), "| 해상도 고유값:", ann[["width","height"]].drop_duplicates().values.tolist())

# 품질 플래그: 파일명 약품수 vs 실제 bbox수
def ncodes(fn): return len([c for c in fn.split("_")[0].split("-")[1:] if c.isdigit()])
codes = per_img.index.to_series().apply(ncodes)
mism = int((codes.values != per_img.values).sum())
print("파일명 약품수 ≠ 실제 bbox수 이미지:", mism)

# SSOT 불변식
class_map = json.loads((SSOT/"class_map.json").read_text(encoding="utf-8"))
drug_master = json.loads((SSOT/"drug_master.json").read_text(encoding="utf-8"))
ok = set(ann.category_id.unique()) <= {int(k) for k in drug_master}
print("class_map num_classes:", class_map["num_classes"], "| drug_master:", len(drug_master), "| 불변식 OK:", ok)""")
)

cells.append(
    md(
        """> **§1 결론 (실행 후 확정)** — 디스크·COCO·불변식이 사전 발견과 일치함을 자동 검증. 핵심: (i) train 232 조합 이미지, 763 pill 객체, 56 클래스; (ii) 무효 bbox 0·해상도 976×1280 단일; (iii) 이미지당 객체 `{2:7,3:151,4:74}`; (iv) 품질 플래그 ~7장(개별 확인 대상, (c)에서); (v) 불변식 OK. → **다음: (b) 클래스 분포·조합 공기출현.**"""
    )
)

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path(
    "/Users/macbook/dev/learning/codeit/01_Proj_HealthEat_cla/project1-3team/beamsearch/LHK/01_eda_domain_anchor.ipynb"
)
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("written:", out, "| cells:", len(cells))
