"""05_rtdetr_final_colab.ipynb 생성 — 번들 → RT-DETR-l 전량학습 → test842 예측 → 제출 CSV."""

import json
from pathlib import Path

OUT = Path(
    "/Users/macbook/dev/personal/claude-personal/multi-machine/projects/healtheat/05_rtdetr_final_colab.ipynb"
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
        "# RT-DETR-l 최종 (Colab) — 앙상블용 다양성 모델",
        "",
        "**목적:** YOLO11s(로컬 0.985)와 **다른 아키텍처(트랜스포머·NMS-free)** RT-DETR-l을 **같은 최고 데이터**",
        "(real232 corrected + synth696 + aihub 7836 인페인트)로 학습 → test842 예측 → 제출 CSV. YOLO와 앙상블.",
        "",
        "**실행:** ① 런타임 GPU(A100 권장, 데이터 큼) ② `rtdetr_final_bundle.zip`을 Drive 업로드 → 경로 지정 ③ 위→아래.",
        "> RT-DETR은 MPS 미지원이라 로컬 불가 → Colab 전용. 데이터 8764장이라 **A100/L4 권장**(T4는 느림·시간초과 위험).",
    ),
    code(
        "from google.colab import drive",
        "drive.mount('/content/drive')",
        "BUNDLE = '/content/drive/MyDrive/rtdetr_final_bundle.zip'  # 실제 경로로 수정",
    ),
    code(
        "import zipfile, torch, os",
        "from pathlib import Path",
        "ROOT = Path('/content/rtdetr_bundle')",
        "if not ROOT.exists():",
        "    with zipfile.ZipFile(BUNDLE) as z: z.extractall(ROOT)",
        "assert torch.cuda.is_available(), 'GPU 런타임 아님!'",
        "print('GPU:', torch.cuda.get_device_name(0))",
        "(ROOT/'data.yaml').write_text((ROOT/'data.yaml').read_text().replace('path: .', f'path: {ROOT}'))",
        "print('train', len(list((ROOT/'images/train').iterdir())), '| test', len(list((ROOT/'test').iterdir())))",
    ),
    code("!pip -q install ultralytics"),
    md(
        "## 1. RT-DETR-l 학습 (전량, 홀드아웃 없음)",
        "**epochs=40**: RT-DETR은 이 데이터서 ~epoch15에 near-peak(mAP50 0.995)로 확인됨 → 40이면 충분(80은 낭비).",
        "batch는 GPU 메모리 따라(A100=8~16, L4=4~8, T4=4).",
    ),
    code(
        "from ultralytics import RTDETR",
        "model = RTDETR('rtdetr-l.pt')",
        "model.train(",
        "    data=str(ROOT/'data.yaml'), epochs=40, imgsz=640, batch=8, device=0, seed=42,",
        "    deterministic=True, optimizer='auto', patience=40, workers=4,",
        "    project=str(ROOT/'runs'), name='rtdetr_final', exist_ok=True, plots=False, verbose=True,",
        ")",
    ),
    md(
        "## 2. test842 예측 → 제출 CSV (canonical)",
        "conf0.001/iou0.6/max_det30, category_id=dl_idx. YOLO 제출과 동일 포맷 → 앙상블·비교 가능.",
    ),
    code(
        "import csv, json, numpy as np",
        "from ultralytics import RTDETR",
        "cm = json.load(open(ROOT/'class_map.json'))",
        "m2c = {int(k): v for k, v in cm['model_index_to_category_id'].items()}",
        "best = RTDETR(str(ROOT/'runs/rtdetr_final/weights/best.pt'))",
        "test = sorted((ROOT/'test').glob('*.png'), key=lambda p: int(p.stem))",
        "rows, aid = [], 1",
        "for p, r in zip(test, best.predict([str(x) for x in test], conf=0.001, iou=0.6, max_det=30, imgsz=640, device=0, verbose=False, stream=True)):",
        "    iid = int(p.stem)",
        "    for b, c, s in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy()):",
        "        x1, y1, x2, y2 = b",
        "        rows.append([aid, iid, m2c[int(c)], round(float(x1),1), round(float(y1),1), round(float(x2-x1),1), round(float(y2-y1),1), round(float(s),4)]); aid += 1",
        "out = ROOT/'submission_rtdetr_final.csv'",
        "with open(out,'w',newline='') as f:",
        "    w = csv.writer(f); w.writerow(['annotation_id','image_id','category_id','bbox_x','bbox_y','bbox_w','bbox_h','score']); w.writerows(rows)",
        "print('rows', len(rows), 'images', len({r[1] for r in rows}), '/842')",
        "from google.colab import files; files.download(str(out))",
    ),
    md(
        "## 3. (선택) 앙상블 재료",
        "이 `submission_rtdetr_final.csv`와 로컬 YOLO11s CSV를 **WBF(Weighted Boxes Fusion)**로 합치면 다양성 이득.",
        "WBF는 로컬에서 두 CSV를 입력으로 실행(별도 스크립트). RT-DETR 단독 점수도 먼저 제출해 확인 권장.",
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
