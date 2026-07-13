"""Test-time predict 튜닝 — 대회지표 mAP@[0.75:0.95]를 최대화하는 추론 설정 탐색.
과적합 방지: 각 config를 3폴드(f0/f1/f2) s696 모델에 대해 각 폴드 val로 평가 → 폴드평균으로 랭킹.
conf=0.001 고정(mAP는 PR곡선 전체), max_det=100(COCO표준). 레버: imgsz·iou(NMS)·agnostic_nms."""

import json
import itertools
import numpy as np
import warnings
import contextlib
import io
from pathlib import Path
from collections import defaultdict
from ultralytics import YOLO
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

warnings.filterwarnings("ignore")
BASE = Path("/Users/macbook/dev/learning/codeit/01_Proj_HealthEat_cla")
TA = BASE / "01_data/01_sprint_ai_project1_data/train_annotations"
TI = BASE / "01_data/01_sprint_ai_project1_data/train_images"
LHK = BASE / "project1-3team/beamsearch/LHK"
KRUNS = LHK / "runs/kfold"
cm = json.load(open(LHK / "data/processed/class_map.json"))
c2m = {int(k): v for k, v in cm["category_id_to_model_index"].items()}
m2c = {int(k): v for k, v in cm["model_index_to_category_id"].items()}
FOLDS = [0, 1, 2]

# GT + 폴드 (prep_yolo 동일 로직)
img_boxes, img_wh = defaultdict(list), {}
for jf in TA.rglob("*.json"):
    d = json.load(open(jf))
    im = d["images"][0]
    img_wh[im["file_name"]] = (im["width"], im["height"])
    for a in d["annotations"]:
        img_boxes[im["file_name"]].append(
            (int(a["category_id"]), [float(v) for v in a["bbox"]])
        )
files = sorted(img_boxes)
sets = {fn: frozenset(c for c, _ in img_boxes[fn]) for fn in files}
combo_of = {c: i for i, c in enumerate(dict.fromkeys(sets[fn] for fn in files))}
groups = np.array([combo_of[sets[fn]] for fn in files])
rng = np.random.default_rng(42)
uniq = np.array(sorted(set(groups)))
rng.shuffle(uniq)
fold_groups = np.array_split(uniq, 5)
g2f = {int(g): k for k, fl in enumerate(fold_groups) for g in fl}
img_fold = {fn: g2f[int(groups[i])] for i, fn in enumerate(files)}


def build_gt(val_files):
    nid = {fn: i + 1 for i, fn in enumerate(val_files)}
    gt = {
        "images": [],
        "annotations": [],
        "categories": [{"id": c} for c in sorted(set(m2c.values()))],
    }
    aid = 1
    for fn in val_files:
        W, H = img_wh[fn]
        gt["images"].append({"id": nid[fn], "file_name": fn, "width": W, "height": H})
        for c, (x, y, w, h) in img_boxes[fn]:
            gt["annotations"].append(
                {
                    "id": aid,
                    "image_id": nid[fn],
                    "category_id": c,
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                }
            )
            aid += 1
    return gt, nid


# config 그리드
IMGSZ = [640, 800, 960, 1280]
IOU = [0.5, 0.6, 0.7]
AGN = [False, True]
configs = list(itertools.product(IMGSZ, IOU, AGN))
results = {c: [] for c in configs}

for f in FOLDS:
    val_files = sorted(fn for fn in files if img_fold[fn] == f)
    gt, nid = build_gt(val_files)
    gp = f"/tmp/tt_gt_f{f}.json"
    json.dump(gt, open(gp, "w"))
    cocoGt = COCO(gp)
    model = YOLO(str(KRUNS / f"f{f}_s696/weights/best.pt"))
    paths = [str(TI / fn) for fn in val_files]
    for sz, iou, agn in configs:
        preds = model.predict(
            paths,
            conf=0.001,
            iou=iou,
            max_det=100,
            imgsz=sz,
            agnostic_nms=agn,
            device="mps",
            verbose=False,
        )
        dts = []
        for fn, res in zip(val_files, preds):
            for b, cl, s in zip(
                res.boxes.xyxy.cpu().numpy(),
                res.boxes.cls.cpu().numpy(),
                res.boxes.conf.cpu().numpy(),
            ):
                x1, y1, x2, y2 = b
                dts.append(
                    {
                        "image_id": nid[fn],
                        "category_id": m2c[int(cl)],
                        "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                        "score": float(s),
                    }
                )
        e = COCOeval(cocoGt, cocoGt.loadRes(dts), "bbox")
        e.params.iouThrs = np.linspace(0.75, 0.95, 5)
        with contextlib.redirect_stdout(io.StringIO()):
            e.evaluate()
            e.accumulate()
            e.summarize()
        results[(sz, iou, agn)].append(float(e.stats[0]))
    print(f"fold{f} 완료 ({len(val_files)} val)", flush=True)

# 랭킹
rows = [(c, float(np.mean(v)), float(np.std(v)), v) for c, v in results.items()]
rows.sort(key=lambda r: -r[1])
base = np.mean(results[(640, 0.6, False)])  # 현재 기본과 유사
print(f"\n현재 유사기준 (imgsz640/iou0.6/agn F) 폴드평균: {base:.4f}\n", flush=True)
print(f"{'imgsz':>6}{'iou':>5}{'agn':>6}{'mean':>9}{'std':>7}   per-fold", flush=True)
for (sz, iou, agn), mean, std, v in rows[:12]:
    mark = "  <= BEST" if (sz, iou, agn) == rows[0][0] else ""
    print(
        f"{sz:>6}{iou:>5}{str(agn):>6}{mean:>9.4f}{std:>7.4f}   {[round(x, 4) for x in v]}{mark}",
        flush=True,
    )
best = rows[0]
json.dump(
    {
        "best_config": {
            "imgsz": best[0][0],
            "iou": best[0][1],
            "agnostic_nms": best[0][2],
            "conf": 0.001,
            "max_det": 100,
        },
        "best_mean": round(best[1], 4),
        "baseline_640_06_F": round(float(base), 4),
        "gain": round(best[1] - float(base), 4),
        "all": [
            {"imgsz": c[0], "iou": c[1], "agn": c[2], "mean": round(m, 4), "folds": v}
            for c, m, s, v in rows
        ],
    },
    open(KRUNS / "ttune.json", "w"),
    indent=2,
)
print(
    f"\n>>> BEST: imgsz{best[0][0]} iou{best[0][1]} agn{best[0][2]} = {best[1]:.4f} (기준 대비 {best[1] - float(base):+.4f})",
    flush=True,
)
print(f"저장: {KRUNS / 'ttune.json'}", flush=True)
