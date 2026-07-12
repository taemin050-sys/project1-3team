"""멀티폴드 GroupKFold 스윕 — 증강 변주를 폴드평균 mAP@[0.75:0.95]로 재랭킹(단일폴드 노이즈 극복).
폴드 정의 = prep_yolo와 동일(클래스-set GroupKFold 5분할, seed=42). val=해당 폴드, train=나머지+synth(train전용).
결과는 runs/kfold/results.jsonl에 append(재개 가능). 하이퍼는 e2/e3/e4와 동일(YOLO11n·50ep·640·b16·mps·seed42·patience20)."""

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
import paths  # 머신 자동 경로 (MacBook/Studio 공용)

TA, TI = paths.TRAIN_ANNOTATIONS, paths.TRAIN_IMAGES
LHK, SSOT, RUNS = paths.LHK, paths.SSOT, paths.RUNS
PROCESSED = paths.PROCESSED
WORK = LHK / "data/kfold_work"
KRUNS = RUNS / "kfold"
KRUNS.mkdir(parents=True, exist_ok=True)
RESULTS = paths.results_file()  # 머신별 결과 파일 → 병합충돌 0
SEED = 42

FOLDS = [0, 1, 2]  # 3폴드(0~2). 필요시 확장.
VARIANTS = {  # 이름 → 추가할 synth 디렉토리(train 전용). base=real만
    "base": [],
    "s696": ["kaggle_sam2_synth_v2_kaggle_696"],
    "s696_1500nat": [
        "kaggle_sam2_synth_v2_kaggle_696",
        "kaggle_sam2_synth_v2_kaggle_1500_696style",
    ],
    "s696_2500bal": [
        "kaggle_sam2_synth_v2_kaggle_696",
        "kaggle_sam2_synth_v2_kaggle_2500",
    ],
}

cm = json.load(open(SSOT / "class_map.json", encoding="utf-8"))
c2m = {int(k): v for k, v in cm["category_id_to_model_index"].items()}
m2c = {int(k): v for k, v in cm["model_index_to_category_id"].items()}
NC = cm["num_classes"]

# ---------- 실데이터 로드 + 폴드 그룹 (prep_yolo 로직 동일) ----------
img_anns, img_wh = defaultdict(list), {}
for jf in TA.rglob("*.json"):
    d = json.load(open(jf, encoding="utf-8"))
    im = d["images"][0]
    fn = im["file_name"]
    img_wh[fn] = (im["width"], im["height"])
    for a in d["annotations"]:
        img_anns[fn].append((int(a["category_id"]), a["bbox"]))
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
W, H = 976, 1280


def build_split(fold, synth_dirs):
    """val=fold 실이미지, train=나머지 실 + synth. WORK 디렉토리 재구성."""
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
            f"{c2m[cid]} {(x + bw / 2) / w:.6f} {(y + bh / 2) / h:.6f} {bw / w:.6f} {bh / h:.6f}"
            for cid, (x, y, bw, bh) in img_anns[fn]
        ]
        (WORK / "labels" / sp / (Path(fn).stem + ".txt")).write_text("\n".join(lines))
        if sp == "val":
            nval += 1
        else:
            ntr += 1
    nsyn = 0
    for j, sd in enumerate(synth_dirs):
        pref = f"s{j}_"
        coco = json.load(open(PROCESSED / sd / "coco/annotations_coco.json"))
        anns_by = defaultdict(list)
        for a in coco["annotations"]:
            anns_by[a["image_id"]].append(a)
        for im in coco["images"]:
            src = PROCESSED / sd / "coco/images" / im["file_name"]
            if not src.exists():
                continue
            name = pref + im["file_name"]
            os.symlink(os.path.realpath(src), WORK / "images/train" / name)
            iw, ih = im["width"], im["height"]
            lines = [
                f"{c2m[int(a['category_id'])]} {(a['bbox'][0] + a['bbox'][2] / 2) / iw:.6f} {(a['bbox'][1] + a['bbox'][3] / 2) / ih:.6f} {a['bbox'][2] / iw:.6f} {a['bbox'][3] / ih:.6f}"
                for a in anns_by[im["id"]]
            ]
            (WORK / "labels/train" / (Path(name).stem + ".txt")).write_text(
                "\n".join(lines)
            )
            nsyn += 1
    names = "\n".join(f"  {i}: '{m2c[i]}'" for i in range(NC))
    (WORK / "data.yaml").write_text(
        f"path: {WORK}\ntrain: images/train\nval: images/val\nnc: {NC}\nnames:\n{names}\n"
    )
    return ntr, nsyn, nval


def evaluate(best, fold):
    val_imgs = sorted((WORK / "images/val").glob("*.png"))
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
        for ln in (WORK / "labels/val" / (p.stem + ".txt")).read_text().splitlines():
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
    gp = KRUNS / f"gt_f{fold}.json"
    json.dump(gt, open(gp, "w"))
    cocoGt = COCO(str(gp))
    dts = []
    for p, res in zip(
        val_imgs,
        best.predict(
            [str(x) for x in val_imgs],
            conf=0.001,
            iou=0.6,
            max_det=100,
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
    return float(e.stats[0])


def _all_result_files():
    return sorted(glob.glob(str(paths.RESULTS_DIR / "*.jsonl"))) + [
        str(KRUNS / "results.jsonl")
    ]


done = set()
for rf in _all_result_files():  # 모든 머신 결과 병합 → 이미 끝난 (variant,fold) skip
    if os.path.exists(rf):
        for ln in open(rf):
            if ln.strip():
                r = json.loads(ln)
                done.add((r["variant"], r["fold"]))

for fold in FOLDS:
    for vname, sdirs in VARIANTS.items():
        if (vname, fold) in done:
            print(f"[skip] {vname} fold{fold} (이미 완료)", flush=True)
            continue
        ntr, nsyn, nval = build_split(fold, sdirs)
        print(
            f"\n=== {vname} fold{fold}: train real{ntr}+synth{nsyn} val{nval} ===",
            flush=True,
        )
        YOLO("yolo11n.pt").train(
            data=str(WORK / "data.yaml"),
            epochs=50,
            imgsz=640,
            batch=16,
            device="mps",
            seed=SEED,
            deterministic=True,
            workers=4,
            patience=20,
            project=str(KRUNS),
            name=f"f{fold}_{vname}",
            exist_ok=True,
            plots=False,
            verbose=False,
        )
        best = YOLO(str(KRUNS / f"f{fold}_{vname}/weights/best.pt"))
        mAP = evaluate(best, fold)
        rec = {
            "variant": vname,
            "fold": fold,
            "mAP_75_95": round(mAP, 4),
            "n_train_real": ntr,
            "n_synth": nsyn,
            "n_val": nval,
            "machine": paths.MACHINE,
        }
        with open(RESULTS, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f">>> {vname} fold{fold}: mAP@[0.75:0.95] = {mAP:.4f}", flush=True)

# ---------- 요약 (모든 머신 결과 병합) ----------
recs = []
for rf in _all_result_files():
    if os.path.exists(rf):
        recs += [json.loads(ln) for ln in open(rf) if ln.strip()]
by_v = defaultdict(list)
for r in recs:
    if r["fold"] in FOLDS:
        by_v[r["variant"]].append(r["mAP_75_95"])
print("\n========== 폴드평균 랭킹 (mAP@[0.75:0.95]) ==========", flush=True)
print(f"{'variant':<16}{'mean':>8}{'std':>8}{'folds':>8}   per-fold", flush=True)
for v in VARIANTS:
    xs = by_v.get(v, [])
    if xs:
        print(
            f"{v:<16}{np.mean(xs):>8.4f}{np.std(xs):>8.4f}{len(xs):>8}   {[round(x, 4) for x in xs]}",
            flush=True,
        )
json.dump(
    {
        v: {
            "mean": round(float(np.mean(by_v[v])), 4),
            "std": round(float(np.std(by_v[v])), 4),
            "folds": by_v[v],
        }
        for v in VARIANTS
        if by_v.get(v)
    },
    open(paths.LAB / "summary.json", "w"),
    indent=2,
)
print(f"\n요약 저장: {paths.LAB / 'summary.json'}", flush=True)
