"""E1 스모크 후처리: (A) 로컬 val mAP@[0.75:0.95] 하니스, (B) test 제출 CSV + 포맷검증. LHK/MPS."""

import json
import csv
from pathlib import Path
import numpy as np
from ultralytics import YOLO
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

BASE = Path("/Users/macbook/dev/learning/codeit/01_Proj_HealthEat_cla")
DATA = BASE / "01_data/01_sprint_ai_project1_data"
LHK = BASE / "project1-3team/beamsearch/LHK"
SSOT = LHK / "data/processed"
YD = LHK / "data/yolo"
RUN = LHK / "runs/e1_smoke"
W, H = 976, 1280

m2c = {
    int(k): v
    for k, v in json.load(open(SSOT / "class_map.json"))[
        "model_index_to_category_id"
    ].items()
}
model = YOLO(str(RUN / "weights/best.pt"))

# ---------- (A) 로컬 val mAP@[0.75:0.95] ----------
val_imgs = sorted((YD / "images/val").glob("*.png"))
nid = {p.name: i + 1 for i, p in enumerate(val_imgs)}
gt = {
    "images": [],
    "annotations": [],
    "categories": [{"id": c} for c in sorted(set(m2c.values()))],
}
aid = 1
for p in val_imgs:
    iid = nid[p.name]
    gt["images"].append({"id": iid, "file_name": p.name, "width": W, "height": H})
    for ln in (YD / "labels/val" / (p.stem + ".txt")).read_text().splitlines():
        if not ln.strip():
            continue
        m, cx, cy, nw, nh = ln.split()
        m = int(m)
        cx, cy, nw, nh = map(float, (cx, cy, nw, nh))
        gt["annotations"].append(
            {
                "id": aid,
                "image_id": iid,
                "category_id": m2c[m],
                "bbox": [(cx - nw / 2) * W, (cy - nh / 2) * H, nw * W, nh * H],
                "area": nw * W * nh * H,
                "iscrowd": 0,
            }
        )
        aid += 1
json.dump(gt, open(RUN / "val_gt.json", "w"))
cocoGt = COCO(str(RUN / "val_gt.json"))

dts = []
for p, res in zip(
    val_imgs,
    model.predict(
        [str(x) for x in val_imgs],
        conf=0.001,
        iou=0.6,
        max_det=20,
        device="mps",
        verbose=False,
    ),
):
    iid = nid[p.name]
    for box, cls, cf in zip(
        res.boxes.xyxy.cpu().numpy(),
        res.boxes.cls.cpu().numpy(),
        res.boxes.conf.cpu().numpy(),
    ):
        x1, y1, x2, y2 = box
        dts.append(
            {
                "image_id": iid,
                "category_id": m2c[int(cls)],
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score": float(cf),
            }
        )
print(f"\nval 예측 박스: {len(dts)}")
if dts:
    ev = COCOeval(cocoGt, cocoGt.loadRes(dts), "bbox")
    ev.params.iouThrs = np.linspace(0.75, 0.95, 5)
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    print(f"\n>>> 로컬 val mAP@[0.75:0.95] = {ev.stats[0]:.4f}")

# ---------- (B) test 제출 CSV ----------
test_imgs = sorted((DATA / "test_images").glob("*.png"), key=lambda p: int(p.stem))
rows = []
aid = 1
for p, res in zip(
    test_imgs,
    model.predict(
        [str(x) for x in test_imgs],
        conf=0.001,
        iou=0.6,
        max_det=10,
        device="mps",
        verbose=False,
        stream=True,
    ),
):
    iid = int(p.stem)
    for box, cls, cf in zip(
        res.boxes.xyxy.cpu().numpy(),
        res.boxes.cls.cpu().numpy(),
        res.boxes.conf.cpu().numpy(),
    ):
        x1, y1, x2, y2 = box
        rows.append(
            [
                aid,
                iid,
                m2c[int(cls)],
                round(float(x1), 1),
                round(float(y1), 1),
                round(float(x2 - x1), 1),
                round(float(y2 - y1), 1),
                round(float(cf), 4),
            ]
        )
        aid += 1
sub = RUN / "submission_e1_smoke.csv"
with open(sub, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(
        [
            "annotation_id",
            "image_id",
            "category_id",
            "bbox_x",
            "bbox_y",
            "bbox_w",
            "bbox_h",
            "score",
        ]
    )
    w.writerows(rows)
imgids = {r[1] for r in rows}
cats = {r[2] for r in rows}
print(f"\n제출 CSV: {sub}")
print(
    f" rows={len(rows)} | images={len(imgids)} (test 842 기대) | cats={len(cats)} | cat⊆train56: {cats <= set(m2c.values())}"
)
