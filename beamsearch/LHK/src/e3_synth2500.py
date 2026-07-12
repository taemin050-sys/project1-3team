"""E3 (LHK/MPS): real181(fold0 train) + synth2500 → ① fold0 val mAP@[0.75:0.95](곡선점) + ② test842 제출 CSV.
곡선: baseline 0.686 → +synth696 0.933 → +synth2500 ?. val 51은 held-out(정직)."""

import json
import os
import csv
import shutil
from pathlib import Path
from collections import defaultdict
import numpy as np
from ultralytics import YOLO
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

BASE = Path("/Users/macbook/dev/learning/codeit/01_Proj_HealthEat_cla")
LHK = BASE / "project1-3team/beamsearch/LHK"
YD, SSOT, RUNS = LHK / "data/yolo", LHK / "data/processed", LHK / "runs"
SYNTH_DIRS = [
    (BASE / "01_data/processed/kaggle_sam2_synth_v2_kaggle_696", "s696_"),
    (BASE / "01_data/processed/kaggle_sam2_synth_v2_kaggle_2500", "s2500_"),
]
TEST = BASE / "01_data/01_sprint_ai_project1_data/test_images"
AUG = LHK / "data/yolo_aug_combined"
W, H = 976, 1280
cm = json.load(open(SSOT / "class_map.json"))
c2m = {int(k): v for k, v in cm["category_id_to_model_index"].items()}
m2c = {int(k): v for k, v in cm["model_index_to_category_id"].items()}

# ---------- 1) yolo_aug2500: train=real181+synth2500, val=fold0 51 ----------
if AUG.exists():
    shutil.rmtree(AUG)
for sp in ("train", "val"):
    (AUG / "images" / sp).mkdir(parents=True)
    (AUG / "labels" / sp).mkdir(parents=True)
for sp in ("train", "val"):
    for p in sorted((YD / "images" / sp).glob("*.png")):
        os.symlink(os.path.realpath(p), AUG / "images" / sp / p.name)
        shutil.copy(
            YD / "labels" / sp / (p.stem + ".txt"),
            AUG / "labels" / sp / (p.stem + ".txt"),
        )
n_real = len(list((AUG / "images/train").glob("*")))
n_syn = 0
for SYNTH, pref in SYNTH_DIRS:  # 696 + 2500 합본 (접두어로 파일명 충돌 방지)
    coco = json.load(open(SYNTH / "coco/annotations_coco.json"))
    anns_by = defaultdict(list)
    for a in coco["annotations"]:
        anns_by[a["image_id"]].append(a)
    for im in coco["images"]:
        src = SYNTH / "coco/images" / im["file_name"]
        if not src.exists():
            continue
        name = pref + im["file_name"]
        os.symlink(os.path.realpath(src), AUG / "images/train" / name)
        iw, ih = im["width"], im["height"]
        lines = [
            f"{c2m[int(a['category_id'])]} {(a['bbox'][0] + a['bbox'][2] / 2) / iw:.6f} {(a['bbox'][1] + a['bbox'][3] / 2) / ih:.6f} {a['bbox'][2] / iw:.6f} {a['bbox'][3] / ih:.6f}"
            for a in anns_by[im["id"]]
        ]
        (AUG / "labels/train" / (Path(name).stem + ".txt")).write_text("\n".join(lines))
        n_syn += 1
names = "\n".join(f"  {i}: '{m2c[i]}'" for i in range(56))
(AUG / "data.yaml").write_text(
    f"path: {AUG}\ntrain: images/train\nval: images/val\nnc: 56\nnames:\n{names}\n"
)
print(f"train real {n_real} + synth {n_syn} = {n_real + n_syn} | val 51", flush=True)

# ---------- 2) 학습 (baseline 동일 설정) ----------
YOLO("yolo11n.pt").train(
    data=str(AUG / "data.yaml"),
    epochs=50,
    imgsz=640,
    batch=16,
    device="mps",
    seed=42,
    deterministic=True,
    workers=4,
    patience=20,
    project=str(RUNS),
    name="e3_combined",
    exist_ok=True,
    plots=False,
    verbose=False,
)
best = YOLO(str(RUNS / "e3_combined/weights/best.pt"))

# ---------- 3) fold0 val mAP@[0.75:0.95] ----------
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
json.dump(gt, open(RUNS / "val_gt.json", "w"))
cocoGt = COCO(str(RUNS / "val_gt.json"))
dts = []
for p, res in zip(
    val_imgs,
    best.predict(
        [str(x) for x in val_imgs],
        conf=0.001,
        iou=0.6,
        max_det=20,
        device="mps",
        verbose=False,
    ),
):
    for b, c, s in zip(
        res.boxes.xyxy.cpu().numpy(),
        res.boxes.cls.cpu().numpy(),
        res.boxes.conf.cpu().numpy(),
    ):
        x1, y1, x2, y2 = b
        dts.append(
            {
                "image_id": nid[p.name],
                "category_id": m2c[int(c)],
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score": float(s),
            }
        )
e = COCOeval(cocoGt, cocoGt.loadRes(dts), "bbox")
e.params.iouThrs = np.linspace(0.75, 0.95, 5)
e.evaluate()
e.accumulate()
e.summarize()
mAP = float(e.stats[0])
print(
    f"\n>>> E3 real+synth(696+2500={n_syn}): fold0 val mAP@[0.75:0.95] = {mAP:.4f}  (곡선: 0.686 → 696:0.933 → 합본:{mAP:.4f})",
    flush=True,
)
json.dump(
    {
        "baseline": 0.686,
        "synth696": 0.9328,
        "synth_combined": round(mAP, 4),
        "train": f"real{n_real}+synth{n_syn}(696+2500)",
    },
    open(RUNS / "e3_curve.json", "w"),
    indent=2,
)

# ---------- 4) test842 제출 CSV ----------
test_imgs = sorted(TEST.glob("*.png"), key=lambda p: int(p.stem))
rows = []
aid = 1
for p, res in zip(
    test_imgs,
    best.predict(
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
    for b, c, s in zip(
        res.boxes.xyxy.cpu().numpy(),
        res.boxes.cls.cpu().numpy(),
        res.boxes.conf.cpu().numpy(),
    ):
        x1, y1, x2, y2 = b
        rows.append(
            [
                aid,
                iid,
                m2c[int(c)],
                round(float(x1), 1),
                round(float(y1), 1),
                round(float(x2 - x1), 1),
                round(float(y2 - y1), 1),
                round(float(s), 4),
            ]
        )
        aid += 1
sub = RUNS / "submission_synth_combined.csv"
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
print(
    f"\n>>> 제출 CSV: {sub} | rows={len(rows)} images={len({r[1] for r in rows})} cats={len({r[2] for r in rows})} cat⊆56={ {r[2] for r in rows} <= set(m2c.values()) }",
    flush=True,
)
