"""04_rtdetr_colab.ipynb 생성: E3와 동일 데이터/fold0 val/mAP 하니스로 RT-DETR 학습 (RT-DETR 특성 반영 튜닝)."""

import json
from pathlib import Path

OUT = Path(
    "/Users/macbook/dev/learning/codeit/01_Proj_HealthEat_cla/project1-3team/beamsearch/LHK/04_rtdetr_colab.ipynb"
)


def md(*s):
    return {"cell_type": "markdown", "metadata": {}, "source": [x + "\n" for x in s]}


def code(*s):
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [x + "\n" for x in s],
    }


cells = [
    md(
        "# RT-DETR (Colab/CUDA) — E3와 **동일 데이터셋** 공정 비교",
        "",
        "**목적:** YOLO11n E3(real181+synth3196, fold0)와 **같은 학습셋·같은 fold0 val·같은 mAP@[0.75:0.95] 하니스**로",
        "RT-DETR-l을 돌려 모델 간 정직 비교. RT-DETR은 MPS 미지원(`grid_sampler_2d_backward`)이라 Colab CUDA에서 실행.",
        "",
        "## 공정성 설계 (무엇을 고정하고 무엇을 튜닝하나)",
        "| 항목 | 처리 | 근거 |",
        "|---|---|---|",
        "| 학습 데이터 | **동일** real181(fold0 train) + synth696 + synth2500 = 3377장 | E3와 1:1 동일 소스 |",
        "| 검증셋 | **동일** fold0 val 51장 (real, held-out) | 누수 없는 정직 기준 |",
        "| 평가 지표 | **동일** mAP@[0.75:0.95], pycocotools, `iouThrs=linspace(0.75,0.95,5)` | 대회 지표 그대로 |",
        "| imgsz | **동일** 640 | 해상도 레버는 별도 페이즈 |",
        "| seed | **동일** 42 | 재현성 |",
        "| epochs | **튜닝** 50→100 | DETR류는 YOLO보다 수렴이 느림 → 동일 epoch는 오히려 RT-DETR에 불리(구조 무시) |",
        "| batch | **튜닝** 16→8 | RT-DETR-l이 무거워 T4/L4 VRAM 대응 |",
        "| optimizer | **auto(AdamW)** | ultralytics가 RT-DETR에 AdamW 자동 선택 — DETR 적합 |",
        "",
        "> 즉 **비교 축(데이터·val·지표)은 완전 고정**하고, 모델 구조상 필요한 학습 하이퍼파라미터만 RT-DETR에 맞게 조정.",
        "",
        "## 실행 순서",
        "1. 런타임 → 런타임 유형 변경 → **GPU (T4/L4/A100)**",
        "2. `rtdetr_data_bundle.zip`(로컬 빌드본)을 Colab에 업로드하거나 Drive에 두고 경로 지정",
        "3. 위→아래 순서로 셀 실행",
    ),
    md(
        "## 0. 번들 로드 (업로드 또는 Drive)",
        "",
        "아래 둘 중 하나. **파일 업로드**가 기본, 대용량이면 **Drive** 권장(~1GB).",
    ),
    code(
        "# (A) 직접 업로드",
        "# from google.colab import files; up = files.upload()  # rtdetr_data_bundle.zip 선택",
        "# BUNDLE_ZIP = 'rtdetr_data_bundle.zip'",
        "",
        "# (B) Google Drive (권장) — Drive에 rtdetr_data_bundle.zip 업로드 후:",
        "from google.colab import drive",
        "drive.mount('/content/drive')",
        "BUNDLE_ZIP = '/content/drive/MyDrive/rtdetr_data_bundle.zip'  # 실제 경로로 수정",
    ),
    code(
        "import zipfile, torch, os",
        "from pathlib import Path",
        "ROOT = Path('/content/rtdetr_bundle')",
        "if not ROOT.exists():",
        "    with zipfile.ZipFile(BUNDLE_ZIP) as z: z.extractall(ROOT)",
        "assert torch.cuda.is_available(), 'GPU 런타임이 아님! 런타임 유형을 GPU로 변경하세요.'",
        "print('GPU:', torch.cuda.get_device_name(0))",
        "# data.yaml의 상대경로(path: .)를 절대경로로 고정",
        "dy = ROOT / 'yolo/data.yaml'",
        "dy.write_text(dy.read_text().replace('path: .', f'path: {ROOT}/yolo'))",
        "ntr = len(list((ROOT/'yolo/images/train').iterdir()))",
        "nva = len(list((ROOT/'yolo/images/val').iterdir()))",
        "print(f'train {ntr} | val {nva}')  # 기대: train 3377 | val 51",
    ),
    code("!pip -q install ultralytics pycocotools"),
    md(
        "## 1. RT-DETR-l 학습",
        "RT-DETR 특성 반영: epochs 100, batch 8, imgsz 640, AdamW(auto). 데이터/val/지표는 E3와 동일.",
    ),
    code(
        "from ultralytics import RTDETR",
        "model = RTDETR('rtdetr-l.pt')  # COCO 사전학습",
        "model.train(",
        "    data=str(ROOT / 'yolo/data.yaml'),",
        "    epochs=100, imgsz=640, batch=8, device=0, seed=42, deterministic=True,",
        "    optimizer='auto', patience=30, workers=2,",
        "    project=str(ROOT / 'runs'), name='rtdetr_combined', exist_ok=True,",
        "    plots=False, verbose=True,",
        ")",
    ),
    md(
        "## 2. fold0 val mAP@[0.75:0.95] — E3와 **동일 하니스**",
        "예측→ model_index→dl_idx 매핑 → COCOeval(iouThrs 0.75~0.95). YOLO11 combined 수치와 직접 비교.",
    ),
    code(
        "import json, numpy as np",
        "from ultralytics import RTDETR",
        "from pycocotools.coco import COCO",
        "from pycocotools.cocoeval import COCOeval",
        "",
        "cm = json.load(open(ROOT / 'harness/class_map.json'))",
        "m2c = {int(k): v for k, v in cm['model_index_to_category_id'].items()}",
        "cocoGt = COCO(str(ROOT / 'harness/val_gt.json'))",
        "val_imgs = sorted(p for p in (ROOT/'yolo/images/val').iterdir() if not p.name.startswith('.'))",
        "nid = {p.name: i + 1 for i, p in enumerate(val_imgs)}",
        "best = RTDETR(str(ROOT / 'runs/rtdetr_combined/weights/best.pt'))",
        "dts = []",
        "for p, res in zip(val_imgs, best.predict([str(x) for x in val_imgs], conf=0.001, iou=0.6, max_det=20, device=0, verbose=False)):",
        "    for b, c, s in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy(), res.boxes.conf.cpu().numpy()):",
        "        x1, y1, x2, y2 = b",
        "        dts.append({'image_id': nid[p.name], 'category_id': m2c[int(c)],",
        "                    'bbox': [float(x1), float(y1), float(x2-x1), float(y2-y1)], 'score': float(s)})",
        "e = COCOeval(cocoGt, cocoGt.loadRes(dts), 'bbox')",
        "e.params.iouThrs = np.linspace(0.75, 0.95, 5)",
        "e.evaluate(); e.accumulate(); e.summarize()",
        "print(f'\\n>>> RT-DETR (real181+synth3196): fold0 val mAP@[0.75:0.95] = {e.stats[0]:.4f}')",
        "print('    비교: YOLO11 baseline 0.686 → +synth696 0.933 → +combined(E3) 결과와 대조')",
    ),
    md(
        "## 3. (선택) test842 제출 CSV",
        "테스트 이미지를 별도로 올린 경우에만. `test_images/`(842장 PNG)를 Colab `/content/test_images`에 두거나 Drive 경로 지정.",
    ),
    code(
        "import csv",
        "TEST = Path('/content/test_images')  # 842장 PNG 경로로 수정 (없으면 이 셀 건너뜀)",
        "if TEST.exists():",
        "    rows, aid = [], 1",
        "    test_imgs = sorted(TEST.glob('*.png'), key=lambda p: int(p.stem))",
        "    for p, res in zip(test_imgs, best.predict([str(x) for x in test_imgs], conf=0.001, iou=0.6, max_det=10, device=0, verbose=False, stream=True)):",
        "        iid = int(p.stem)",
        "        for b, c, s in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy(), res.boxes.conf.cpu().numpy()):",
        "            x1, y1, x2, y2 = b",
        "            rows.append([aid, iid, m2c[int(c)], round(float(x1),1), round(float(y1),1), round(float(x2-x1),1), round(float(y2-y1),1), round(float(s),4)]); aid += 1",
        "    out = ROOT / 'runs/submission_rtdetr_combined.csv'",
        "    with open(out, 'w', newline='') as f:",
        "        w = csv.writer(f); w.writerow(['annotation_id','image_id','category_id','bbox_x','bbox_y','bbox_w','bbox_h','score']); w.writerows(rows)",
        "    from google.colab import files; files.download(str(out))",
        "    print('제출 CSV:', out, '| rows', len(rows))",
        "else:",
        "    print('test_images 없음 → fold0 비교만 수행 (제출 CSV 스킵)')",
    ),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "accelerator": "GPU",
        "colab": {"provenance": []},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}
OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
print("노트북:", OUT)
