"""E2 애블레이션 (LHK/MPS): baseline(real 181) vs +synth(real 181 + Codex synth 696).
동일 fold0 val(51 real) · 동일 mAP@[0.75:0.95] 하니스. baseline=e1_baseline(0.686) 참조."""

import json
import os
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
SYNTH = BASE / "01_data/processed/kaggle_sam2_synth_v2_kaggle_696"
AUG = LHK / "data/yolo_aug"
W, H = 976, 1280
cm = json.load(open(SSOT / "class_map.json"))
c2m = {int(k): v for k, v in cm["category_id_to_model_index"].items()}
m2c = {int(k): v for k, v in cm["model_index_to_category_id"].items()}

# ---------- 1) yolo_aug 구성: train=real181+synth696, val=fold0 51 real ----------
if AUG.exists():
    shutil.rmtree(AUG)
for sp in ("train", "val"):
    (AUG / "images" / sp).mkdir(parents=True)
    (AUG / "labels" / sp).mkdir(parents=True)
# real fold0 (symlink 이미지 + 라벨 복사)
for sp in ("train", "val"):
    for p in sorted((YD / "images" / sp).glob("*.png")):
        os.symlink(os.path.realpath(p), AUG / "images" / sp / p.name)
        shutil.copy(
            YD / "labels" / sp / (p.stem + ".txt"),
            AUG / "labels" / sp / (p.stem + ".txt"),
        )
n_real = len(list((AUG / "images/train").glob("*")))
# synth → train
coco = json.load(open(SYNTH / "coco/annotations_coco.json"))
anns_by = defaultdict(list)
for a in coco["annotations"]:
    anns_by[a["image_id"]].append(a)
n_syn = 0
for im in coco["images"]:
    src = SYNTH / "coco/images" / im["file_name"]
    if not src.exists():
        continue
    os.symlink(os.path.realpath(src), AUG / "images/train" / im["file_name"])
    iw, ih = im["width"], im["height"]
    lines = []
    for a in anns_by[im["id"]]:
        x, y, bw, bh = a["bbox"]
        m = c2m[int(a["category_id"])]
        lines.append(
            f"{m} {(x + bw / 2) / iw:.6f} {(y + bh / 2) / ih:.6f} {bw / iw:.6f} {bh / ih:.6f}"
        )
    (AUG / "labels/train" / (Path(im["file_name"]).stem + ".txt")).write_text(
        "\n".join(lines)
    )
    n_syn += 1
names = "\n".join(f"  {i}: '{m2c[i]}'" for i in range(56))
(AUG / "data.yaml").write_text(
    f"path: {AUG}\ntrain: images/train\nval: images/val\nnc: 56\nnames:\n{names}\n"
)
print(
    f"yolo_aug: train real {n_real} + synth {n_syn} = {n_real + n_syn} | val {len(list((AUG / 'images/val').glob('*')))}",
    flush=True,
)

# ---------- 2) 학습 (baseline과 동일 설정) ----------
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
    name="e2_synth",
    exist_ok=True,
    plots=False,
    verbose=False,
)
best = YOLO(str(RUNS / "e2_synth/weights/best.pt"))

# ---------- 3) fold0 val mAP@[0.75:0.95] (동일 하니스) ----------
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
    f"\n>>> E2 +synth(696): 로컬 val mAP@[0.75:0.95] = {mAP:.4f}  (baseline real-only = 0.686)",
    flush=True,
)
print(f">>> Δ = {mAP - 0.686:+.4f}", flush=True)
json.dump(
    {
        "baseline_real": 0.686,
        "plus_synth696": round(mAP, 4),
        "delta": round(mAP - 0.686, 4),
        "train": f"real {n_real} + synth {n_syn}",
    },
    open(RUNS / "e2_ablation.json", "w"),
    indent=2,
)
