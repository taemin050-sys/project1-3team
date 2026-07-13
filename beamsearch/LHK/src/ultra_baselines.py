"""ultralytics 다중 모델 베이스라인 (LHK/MPS): YOLO26 · RT-DETR. 동일 fold0·동일 mAP@[0.75:0.95] 하니스."""

import json
from pathlib import Path
import numpy as np
from ultralytics import YOLO, RTDETR
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

BASE = Path("/Users/macbook/dev/learning/codeit/01_Proj_HealthEat_cla")
LHK = BASE / "project1-3team/beamsearch/LHK"
SSOT, YD, RUNS = LHK / "data/processed", LHK / "data/yolo", LHK / "runs"
W, H = 976, 1280
m2c = {
    int(k): v
    for k, v in json.load(open(SSOT / "class_map.json"))[
        "model_index_to_category_id"
    ].items()
}

# ---- val GT (once) ----
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
gtf = RUNS / "val_gt.json"
json.dump(gt, open(gtf, "w"))
cocoGt = COCO(str(gtf))


def eval_model(model):
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
    if not dts:
        return 0.0, 0.0, 0
    ev = COCOeval(cocoGt, cocoGt.loadRes(dts), "bbox")
    ev.params.iouThrs = np.linspace(0.75, 0.95, 5)
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    ev2 = COCOeval(cocoGt, cocoGt.loadRes(dts), "bbox")
    ev2.params.iouThrs = np.array([0.75])
    ev2.evaluate()
    ev2.accumulate()
    ev2.summarize()
    return float(ev.stats[0]), float(ev2.stats[0]), len(dts)


resf = RUNS / "baselines.json"
results = json.loads(resf.read_text()) if resf.exists() else []
have = {r["model"] for r in results}
if "YOLO11n" not in have:
    results.append(
        {
            "model": "YOLO11n",
            "params_M": 2.6,
            "mAP_75_95": 0.686,
            "ap75": 0.710,
            "epochs": 50,
            "imgsz": 640,
            "note": "E1 reference",
        }
    )

JOBS = [
    ("yolo26n.pt", YOLO, "b_yolo26", "YOLO26n", 16),
    ("rtdetr-l.pt", RTDETR, "b_rtdetr", "RT-DETR-l", 8),
]
for weights, cls, run, label, bs in JOBS:
    if label in have:
        continue
    print(f"\n===== TRAIN {label} =====")
    m = cls(weights)
    m.train(
        data=str(YD / "data.yaml"),
        epochs=50,
        imgsz=640,
        batch=bs,
        device="mps",
        seed=42,
        deterministic=True,
        workers=4,
        patience=20,
        project=str(RUNS),
        name=run,
        exist_ok=True,
        plots=False,
        verbose=False,
    )
    best = cls(str(RUNS / run / "weights/best.pt"))
    pm = sum(p.numel() for p in best.model.parameters()) / 1e6
    mAP, ap75, ndet = eval_model(best)
    results.append(
        {
            "model": label,
            "params_M": round(pm, 1),
            "mAP_75_95": round(mAP, 4),
            "ap75": round(ap75, 4),
            "epochs": 50,
            "imgsz": 640,
            "val_det": ndet,
        }
    )
    resf.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f">>> {label}: mAP@[0.75:0.95]={mAP:.4f} AP@0.75={ap75:.4f} params={pm:.1f}M")

print("\n===== 베이스라인 비교표 (fold0 val) =====")
print(f"{'model':<12}{'params_M':>9}{'mAP@.75:.95':>13}{'AP@.75':>9}")
for r in sorted(results, key=lambda x: -x["mAP_75_95"]):
    print(
        f"{r['model']:<12}{r['params_M']:>9}{r['mAP_75_95']:>13}{r.get('ap75', 0):>9}"
    )
