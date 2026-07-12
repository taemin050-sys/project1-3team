"""06_rtdetr_cover_colab.ipynb 생성 — 클린 커버리지(118) RT-DETR, Colab 중단→재개 안전.
핵심: 체크포인트·결과를 마운트된 구글 드라이브에 저장(project=Drive) → last.pt 매 에폭 갱신 →
세션/사용량 한도로 끊겨도 재접속 후 셀 재실행하면 resume=True로 이어서 학습."""

import json
from pathlib import Path

OUT = Path(
    "/Users/macbook/dev/personal/claude-personal/multi-machine/projects/healtheat/06_rtdetr_cover_colab.ipynb"
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
        "# RT-DETR-l 클린 커버리지(118) — Colab 중단→재개 안전판",
        "",
        "**목적:** 클린 커버리지 데이터(real232 + synth696 + aihub_cover116 − 의심485)로 RT-DETR-l 1회 학습 → test842 예측.",
        "**안전장치:** 가중치·상태를 **마운트된 구글 드라이브**에 저장(`project=Drive`). `last.pt`가 매 에폭 갱신되고",
        "`best.pt`는 갱신 시 저장 → 세션/사용량 한도로 끊겨도 **재접속 후 모든 셀 위→아래 재실행**하면 `resume=True`로 이어서.",
        "",
        "**실행:** ① GPU 런타임(A100/L4 권장) ② `rtdetr_cover_clean_bundle.zip`을 내 드라이브에 업로드 → 아래 경로 확인 ③ 위→아래.",
        "**끊겼을 때:** 런타임 재연결(또는 한도 해제 후) → **같은 노트북 셀을 처음부터 다시 실행**만 하면 자동 이어서 학습.",
    ),
    code(
        "from google.colab import drive",
        "drive.mount('/content/drive')",
        "# ▼ 경로 (필요시 수정)",
        "BUNDLE    = '/content/drive/MyDrive/rtdetr_cover_clean_bundle.zip'  # 업로드한 클린 번들",
        "DRIVE_RUN = '/content/drive/MyDrive/rtdetr_cover_run'              # 체크포인트·결과 저장(영구, 자동 생성)",
    ),
    code(
        "import zipfile, torch, os",
        "from pathlib import Path",
        "# 데이터는 /content(빠름·휘발성)에 풀되, 경로는 항상 동일 → resume 시 data 경로 일치",
        "ROOT = Path('/content/rtdetr_bundle')",
        "if not (ROOT/'data.yaml').exists():",
        "    with zipfile.ZipFile(BUNDLE) as z: z.extractall(ROOT)",
        "    y = (ROOT/'data.yaml').read_text()",
        "    if 'path: .' in y: (ROOT/'data.yaml').write_text(y.replace('path: .', f'path: {ROOT}'))",
        "assert torch.cuda.is_available(), 'GPU 런타임 아님!'",
        "Path(DRIVE_RUN).mkdir(parents=True, exist_ok=True)",
        "print('GPU:', torch.cuda.get_device_name(0))",
        "print('train', len(list((ROOT/'images/train').iterdir())), '| test', len(list((ROOT/'test').iterdir())))",
        "print('nc:', [l for l in (ROOT/'data.yaml').read_text().splitlines() if l.startswith('nc:')])",
    ),
    code("!pip -q install ultralytics"),
    md(
        "## 1. 학습 (Drive에 체크포인트 저장 · 자동 resume)",
        "- `project=DRIVE_RUN` → `weights/last.pt`·`best.pt`가 **드라이브에 매 에폭 저장**(중단 대비).",
        "- 드라이브에 `last.pt`가 있으면 **자동으로 resume**(이어서), 없으면 신규 시작.",
        "- **epochs=40**: 이 데이터서 RT-DETR ~epoch15 near-peak 확인 → 40이면 충분.",
        "- batch: A100=8~16, L4=4~8, T4=4.",
    ),
    code(
        "from pathlib import Path",
        "from ultralytics import RTDETR",
        "RUN = Path(DRIVE_RUN)/'rtdetr_cover'",
        "last = RUN/'weights'/'last.pt'",
        "if last.exists():",
        "    print('▶ 이어서 학습(resume):', last)",
        "    model = RTDETR(str(last))",
        "    model.train(resume=True)   # data·epochs 등은 체크포인트에 저장돼 있어 재지정 불필요",
        "else:",
        "    print('▶ 신규 학습 시작')",
        "    model = RTDETR('rtdetr-l.pt')",
        "    model.train(",
        "        data=str(ROOT/'data.yaml'), epochs=40, imgsz=640, batch=8, device=0, seed=42,",
        "        deterministic=True, optimizer='auto', patience=40, workers=4,",
        "        project=str(DRIVE_RUN), name='rtdetr_cover', exist_ok=True, plots=False, verbose=True,",
        "        # last.pt/best.pt는 기본적으로 매 에폭 저장됨(save_period=-1). Drive에 직접 기록되어 중단 안전.",
        "    )",
        "print('학습 완료 — best.pt:', RUN/'weights'/'best.pt')",
    ),
    md(
        "## 2. test842 예측 → 제출 CSV (Drive 저장)",
        "conf0.001/iou0.6/max_det30, category_id=dl_idx(118). YOLO 커버리지 CSV와 동일 포맷 → WBF 앙상블 가능.",
    ),
    code(
        "import csv, json, torch",
        "from ultralytics import RTDETR",
        "cm = json.load(open(ROOT/'class_map.json'))",
        "m2c = {int(k): v for k, v in cm['model_index_to_category_id'].items()}",
        "best = RTDETR(str(Path(DRIVE_RUN)/'rtdetr_cover/weights/best.pt'))",
        "test = sorted((ROOT/'test').glob('*.png'), key=lambda p: int(p.stem))",
        "rows, aid = [], 1",
        "# per-image 예측: 전체 리스트를 한 배치로 넘기면 대형 test셋에서 CUDA OOM → 1장씩 처리로 회피",
        "for i, p in enumerate(test):",
        "    iid = int(p.stem)",
        "    r = best.predict(str(p), conf=0.001, iou=0.6, max_det=30, imgsz=640, device=0, verbose=False)[0]",
        "    for b, c, s in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy()):",
        "        x1, y1, x2, y2 = b",
        "        rows.append([aid, iid, m2c[int(c)], round(float(x1),1), round(float(y1),1), round(float(x2-x1),1), round(float(y2-y1),1), round(float(s),4)]); aid += 1",
        "    if i % 200 == 0: torch.cuda.empty_cache()",
        "out = Path(DRIVE_RUN)/'submission_rtdetr_cover_clean.csv'",
        "with open(out,'w',newline='') as f:",
        "    w = csv.writer(f); w.writerow(['annotation_id','image_id','category_id','bbox_x','bbox_y','bbox_w','bbox_h','score']); w.writerows(rows)",
        "print('저장(Drive):', out, '| rows', len(rows), '| images', len({r[1] for r in rows}), '/842')",
        "from google.colab import files; files.download(str(out))",
    ),
    md(
        "## 재개 방법 (중단 시)",
        "1. 런타임 재연결(또는 사용량 한도 해제 후 새 런타임).",
        "2. **모든 셀을 위→아래로 다시 실행** — Drive 마운트 → 번들 재압축해제(/content) → 학습 셀이 `last.pt` 감지 → 자동 이어서.",
        "3. 진행 상황·가중치는 `DRIVE_RUN`(내 드라이브 `rtdetr_cover_run/`)에 계속 쌓여 안전.",
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
