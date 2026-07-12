"""E4 (LHK/MPS): 단일변수 실험 — real181 + synth696 + synth1500(696식 자연분포).
목적: '균형강제(2500)' vs '순수 물량증가(696식 1500)' 분리. 2500이 0.9173으로 하락했는데,
같은 물량대를 696식(자연분포)으로 채우면? 유지/개선이면 하락은 균형강제 탓, 동반하락이면 비율/희석 탓.
하이퍼는 e2/e3와 100% 동일(YOLO11n·50ep·640·b16·mps·seed42·patience20). 유일 변수=데이터."""

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
    (BASE / "01_data/processed/kaggle_sam2_synth_v2_kaggle_1500_696style", "s1500_"),
]
TEST = BASE / "01_data/01_sprint_ai_project1_data/test_images"
AUG = LHK / "data/yolo_aug_696style"
W, H = 976, 1280
cm = json.load(open(SSOT / "class_map.json"))
c2m = {int(k): v for k, v in cm["category_id_to_model_index"].items()}
m2c = {int(k): v for k, v in cm["model_index_to_category_id"].items()}

# ---------- 1) yolo_aug_696style: train=real181 + synth696 + synth1500, val=fold0 51 ----------
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
for SYNTH, pref in SYNTH_DIRS:  # 696 + 1500(696식) 합본, 접두어로 파일명 충돌 방지
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
print(
    f"train real {n_real} + synth {n_syn}(696+1500식) = {n_real + n_syn} | val 51 | real:synth=1:{n_syn / n_real:.1f}",
    flush=True,
)

# ---------- 2) 학습 (e2/e3 동일 설정) ----------
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
    name="e4_696style",
    exist_ok=True,
    plots=False,
    verbose=False,
)
best = YOLO(str(RUNS / "e4_696style/weights/best.pt"))

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
    f"\n>>> E4 real+synth696+1500(696식): fold0 val mAP@[0.75:0.95] = {mAP:.4f}",
    flush=True,
)
print(
    f">>> 곡선: baseline 0.686 → 696:0.933 → +2500(균형강제):0.9173 → +1500(696식자연):{mAP:.4f}",
    flush=True,
)
print(
    f">>> 판정: {'균형강제가 원인(자연분포는 유지/개선)' if mAP >= 0.930 else '비율/희석도 작용(자연분포도 하락)' if mAP < 0.925 else '중간(추가 폴드 필요)'}",
    flush=True,
)
json.dump(
    {
        "baseline": 0.686,
        "synth696": 0.9328,
        "synth696_2500_balanced": 0.9173,
        "synth696_1500_natural": round(mAP, 4),
        "train": f"real{n_real}+synth{n_syn}(696+1500식)",
        "ratio": f"1:{n_syn / n_real:.1f}",
    },
    open(RUNS / "e4_curve.json", "w"),
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
sub = RUNS / "submission_696style_1500.csv"
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
    f"\n>>> 제출 CSV: {sub} | rows={len(rows)} images={len({r[1] for r in rows})} cats={len({r[2] for r in rows})}",
    flush=True,
)
