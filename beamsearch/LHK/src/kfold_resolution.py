"""학습 해상도 실험 — s696 데이터를 imgsz 640 vs 960 학습(+동일해상도 추론)으로 3폴드 비교.
동기: test-time 튜닝서 '640학습→고해상추론'은 하락 → 해상도 이득은 학습해상도로 가야 함(976×1280 원본).
결과 variant='s696_r960' → lab-kit 버스(paths→labkit)에 기록. 비교대상 s696(=r640)=0.905."""

import json
import os
import sys
import shutil
from pathlib import Path
from collections import defaultdict
import numpy as np
from ultralytics import YOLO
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths  # 경로 어댑터 (labkit을 sys.path에 올림)
import labkit

TA, TI, PROCESSED = paths.TRAIN_ANNOTATIONS, paths.TRAIN_IMAGES, paths.PROCESSED
SSOT, RUNS = paths.SSOT, paths.RUNS
WORK = paths.LHK / "data/kfold_work_res"
KRUNS = RUNS / "kfold"
KRUNS.mkdir(parents=True, exist_ok=True)
SEED, IMGSZ, VARIANT = 42, 960, "s696_r960"
FOLDS = [0, 1, 2]
SYNTH = ["kaggle_sam2_synth_v2_kaggle_696"]
W, H = 976, 1280

cm = json.load(open(SSOT / "class_map.json", encoding="utf-8"))
c2m = {int(k): v for k, v in cm["category_id_to_model_index"].items()}
m2c = {int(k): v for k, v in cm["model_index_to_category_id"].items()}
NC = cm["num_classes"]

img_anns, img_wh = defaultdict(list), {}
for jf in TA.rglob("*.json"):
    d = json.load(open(jf, encoding="utf-8"))
    im = d["images"][0]
    img_wh[im["file_name"]] = (im["width"], im["height"])
    for a in d["annotations"]:
        img_anns[im["file_name"]].append((int(a["category_id"]), a["bbox"]))
files = sorted(img_anns)
sets = {fn: frozenset(c for c, _ in img_anns[fn]) for fn in files}
combo_of = {c: i for i, c in enumerate(dict.fromkeys(sets[fn] for fn in files))}
groups = np.array([combo_of[sets[fn]] for fn in files])
rng = np.random.default_rng(SEED)
uniq = np.array(sorted(set(groups)))
rng.shuffle(uniq)
fold_groups = np.array_split(uniq, 5)
g2f = {int(g): k for k, fl in enumerate(fold_groups) for g in fl}
img_fold = {fn: g2f[int(groups[i])] for i, fn in enumerate(files)}


def build_split(fold):
    if WORK.exists():
        shutil.rmtree(WORK)
    for sp in ("train", "val"):
        (WORK / "images" / sp).mkdir(parents=True)
        (WORK / "labels" / sp).mkdir(parents=True)
    ntr = nval = 0
    for fn in files:
        sp = "val" if img_fold[fn] == fold else "train"
        w, h = img_wh[fn]
        os.symlink(TI / fn, WORK / "images" / sp / fn)
        lines = [
            f"{c2m[c]} {(x + bw / 2) / w:.6f} {(y + bh / 2) / h:.6f} {bw / w:.6f} {bh / h:.6f}"
            for c, (x, y, bw, bh) in img_anns[fn]
        ]
        (WORK / "labels" / sp / (Path(fn).stem + ".txt")).write_text("\n".join(lines))
        nval += sp == "val"
        ntr += sp == "train"
    nsyn = 0
    for j, sd in enumerate(SYNTH):
        coco = json.load(open(PROCESSED / sd / "coco/annotations_coco.json"))
        anns_by = defaultdict(list)
        for a in coco["annotations"]:
            anns_by[a["image_id"]].append(a)
        for im in coco["images"]:
            src = PROCESSED / sd / "coco/images" / im["file_name"]
            if not src.exists():
                continue
            name = f"s{j}_" + im["file_name"]
            os.symlink(os.path.realpath(src), WORK / "images/train" / name)
            iw, ih = im["width"], im["height"]
            (WORK / "labels/train" / (Path(name).stem + ".txt")).write_text(
                "\n".join(
                    f"{c2m[int(a['category_id'])]} {(a['bbox'][0] + a['bbox'][2] / 2) / iw:.6f} {(a['bbox'][1] + a['bbox'][3] / 2) / ih:.6f} {a['bbox'][2] / iw:.6f} {a['bbox'][3] / ih:.6f}"
                    for a in anns_by[im["id"]]
                )
            )
            nsyn += 1
    names = "\n".join(f"  {i}: '{m2c[i]}'" for i in range(NC))
    (WORK / "data.yaml").write_text(
        f"path: {WORK}\ntrain: images/train\nval: images/val\nnc: {NC}\nnames:\n{names}\n"
    )
    return ntr, nsyn, nval


def evaluate(best, fold):
    val_imgs = sorted(fn for fn in files if img_fold[fn] == fold)
    nid = {fn: i + 1 for i, fn in enumerate(val_imgs)}
    gt = {
        "images": [],
        "annotations": [],
        "categories": [{"id": c} for c in sorted(set(m2c.values()))],
    }
    aid = 1
    for fn in val_imgs:
        gt["images"].append({"id": nid[fn], "file_name": fn, "width": W, "height": H})
        for c, (x, y, w, h) in img_anns[fn]:
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
    gp = KRUNS / f"gt_res_f{fold}.json"
    json.dump(gt, open(gp, "w"))
    cocoGt = COCO(str(gp))
    paths_ = [str(TI / fn) for fn in val_imgs]
    dts = []
    for fn, res in zip(
        val_imgs,
        best.predict(
            paths_,
            conf=0.001,
            iou=0.6,
            max_det=100,
            imgsz=IMGSZ,
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
                    "image_id": nid[fn],
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
    return float(e.stats[0])


done = labkit.done_keys("healtheat", ["variant", "fold"])
for fold in FOLDS:
    if (VARIANT, fold) in done:
        print(f"[skip] {VARIANT} fold{fold}", flush=True)
        continue
    ntr, nsyn, nval = build_split(fold)
    print(
        f"\n=== {VARIANT} fold{fold}: real{ntr}+synth{nsyn} val{nval} @imgsz{IMGSZ} ===",
        flush=True,
    )
    YOLO("yolo11n.pt").train(
        data=str(WORK / "data.yaml"),
        epochs=50,
        imgsz=IMGSZ,
        batch=16,
        device="mps",
        seed=SEED,
        deterministic=True,
        workers=4,
        patience=20,
        project=str(KRUNS),
        name=f"f{fold}_{VARIANT}",
        exist_ok=True,
        plots=False,
        verbose=False,
    )
    best = YOLO(str(KRUNS / f"f{fold}_{VARIANT}/weights/best.pt"))
    mAP = evaluate(best, fold)
    labkit.record(
        "healtheat",
        {
            "variant": VARIANT,
            "fold": fold,
            "mAP_75_95": round(mAP, 4),
            "n_train_real": ntr,
            "n_synth": nsyn,
            "n_val": nval,
            "imgsz": IMGSZ,
        },
    )
    print(f">>> {VARIANT} fold{fold}: mAP@[0.75:0.95] = {mAP:.4f}", flush=True)

recs = [
    r for r in labkit.load_records("healtheat") if r["variant"] in (VARIANT, "s696")
]
byv = defaultdict(list)
for r in recs:
    byv[r["variant"]].append(r["mAP_75_95"])
print("\n===== 해상도 비교 (폴드평균) =====", flush=True)
for v in ("s696", VARIANT):
    if byv.get(v):
        print(
            f"  {v:<12} ({'640' if v == 's696' else IMGSZ}): {np.mean(byv[v]):.4f}  {[round(x, 4) for x in byv[v]]}",
            flush=True,
        )
