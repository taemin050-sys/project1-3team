import json
from pathlib import Path

LHK = Path(
    "/Users/macbook/dev/learning/codeit/01_Proj_HealthEat_cla/project1-3team/beamsearch/LHK"
)


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
    md("""# 📊 모델 베이스라인 비교 — 대회 기본 데이터 (fold0)

> **LHK / Model Architect** · 여러 후보 모델의 베이스라인을 **동일 데이터 · 동일 fold0 · 동일 mAP@[0.75:0.95] 하니스**로 비교.

**목적.** 캐글 최고점 경쟁이 아니라 **체계적·공정한 모델 비교 과정**을 기록한다(채점 기준 = 과정의 논리성·공학적 접근·협업). docs/02 §5.2 후보 지도의 실측 대응.""")
)

cells.append(
    md("""## 방법론 & 디바이스 분리

- **동일 조건:** fold0(조합 단위 GroupKFold, seed=42) · 로컬 mAP@[0.75:0.95](pycocotools, IoU=linspace(0.75,0.95,5)) · 예측 `model_index → dl_idx`(class_map) → `val_gt.json` 채점.
- **MPS 로컬(LHK 정책):** YOLO11n · YOLO26n · Faster R-CNN · RetinaNet · FCOS.
- **CUDA 노트북(Colab/Runpod):** RT-DETR(=`grid_sampler_2d_backward` MPS 미지원) · Cascade R-CNN · DINO/Co-DETR · DINOv2-frozen — 동일 fold0 번들로 별도 실행 후 이 표에 합류.
- ⚠️ **고정예산 스냅샷(무튜닝):** YOLO 50ep@640 · torchvision 20ep@min512. 프레임워크·예산이 달라 **"아키텍처 우열 판정"이 아니라 baseline floor 비교**로 읽어야 한다.""")
)

cells.append(
    code("""import json
from pathlib import Path
import pandas as pd
import matplotlib as mpl, matplotlib.pyplot as plt
mpl.rcParams["font.family"] = "AppleGothic"; mpl.rcParams["axes.unicode_minus"] = False

RUNS = Path("/Users/macbook/dev/learning/codeit/01_Proj_HealthEat_cla/project1-3team/beamsearch/LHK/runs")
df = pd.DataFrame(json.load(open(RUNS / "baselines.json"))).sort_values("mAP_75_95", ascending=False).reset_index(drop=True)
cols = [c for c in ["model", "params_M", "mAP_75_95", "ap75", "epochs", "imgsz", "val_det"] if c in df.columns]
print(df[cols].to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.bar(df.model, df.mAP_75_95, color=["#4C78A8" if m.startswith("YOLO") else "#E45756" for m in df.model])
ax.set(title="MPS 베이스라인 — mAP@[0.75:0.95] (fold0 val)", ylabel="mAP@[0.75:0.95]")
ax.tick_params(axis="x", rotation=20)
for b, v in zip(bars, df.mAP_75_95): ax.text(b.get_x()+b.get_width()/2, v+0.01, f"{v:.3f}", ha="center", fontsize=9)
plt.tight_layout(); plt.show()""")
)

cells.append(
    md("""> **해석**
>
> - **YOLO11n(0.686)이 명확한 1위** — 최소 파라미터(2.6M)로 최고. ultralytics 파이프라인이 이 조건에서 가장 turnkey.
> - **RetinaNet(0.575)** 준수 · **YOLO26n(0.480)** — NMS-free E2E 헤드가 소데이터·고정예산선 **수렴이 더 더딤**(정직한 관찰) · **FCOS(0.186)** 약함(에폭·튜닝 필요).
> - **FasterRCNN-R50 = 0.0 → 학습 발산(버그 아님).** 근거: **val 검출수 0**(임계 0.001에서도 박스 0). 같은 하니스로 RetinaNet(5955)·FCOS(5100)는 정상 검출 → **평가·offset 문제 배제 확정**. 헤드 재초기화 + **LR warmup 없는 SGD lr=0.005** → 초기 발산의 전형. **warmup·lr↓·grad-clip로 재실행 예정.**
>
> **공정성 캐비앗:** 프레임워크·예산·튜닝이 균일하지 않은 **baseline 스냅샷**이다. torchvision 계열은 예산·튜닝에서 불리했고 FasterRCNN은 설정 이슈로 발산했다. 이 표의 가치는 **동일 하니스 비교 + 실패까지 정직히 기록**한 점이다(고도화 대상 선별의 출발점).""")
)

cells.append(
    md("""## CUDA 트랙 (진행 예정)

동일 fold0 번들(`lhk_cuda_bundle.zip`)로 Colab/Runpod에서 실행 후 이 표에 합류:

| 모델 | 근거 | 상태 |
| --- | --- | --- |
| RT-DETR | 실시간 트랜스포머(서빙 대안), MPS 학습 불가 | 대기 |
| Cascade R-CNN | 고-IoU 설계 정합 2-stage(FasterRCNN warmup본과 대조) | 대기 |
| DINO / Co-DETR | 정확도 트랜스포머(deformable-attention) | 대기 |
| DINOv2-frozen | 저데이터 SSL 백본(과적합 억제) | 대기(선택) |

**다음:** ① FasterRCNN warmup 재실행(2-stage 진짜 수치) · ② CUDA 노트북 작성·실행 · ③ 결과 합류 후 고도화 대상(단일→조합 증강) 선정.""")
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
out = LHK / "02_model_baselines.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("written:", out, "| cells:", len(cells))
