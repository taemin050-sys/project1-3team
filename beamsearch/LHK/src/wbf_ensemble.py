"""WBF(Weighted Boxes Fusion) 앙상블 하네스 — 오류분석이 지목한 '박스 로컬라이제이션' 레버.
여러 모델의 박스를 IoU 클러스터링 후 신뢰도 가중평균 → 박스가 타이트해져 고IoU(0.75:0.95) 이득.

3 모드 (WBF_MODE):
  submit   : 제출 CSV N개 융합 → 앙상블 제출 CSV.       (Kaggle 판정용, 기본 경로 a)
  foldpred : 모델 best.pt → fold0-val 예측 CSV 생성.    (eval 재료, YOLO 전용)
  eval     : fold0 예측 CSV N개 융합 → pycocotools 0.75:0.95 채점 (+가중치 그리드). (로컬 검증 경로 b)

공통 env: WBF_IOU(0.6) WBF_SKIP(0.0001) WBF_CONF(avg)
  submit/eval: WBF_INPUTS="csv1:w1,csv2:w2,.."  WBF_OUT(제출 CSV 경로, submit만)
  foldpred   : WBF_MODEL=best.pt 경로  WBF_OUT=출력 CSV
  eval       : WBF_GRID=1 이면 가중치·IoU 그리드 탐색
필요: pip install ensemble-boxes"""

import csv
import json
import os
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
from ensemble_boxes import weighted_boxes_fusion

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths

MODE = os.environ.get("WBF_MODE", "submit")
IOU = float(os.environ.get("WBF_IOU", "0.6"))
SKIP = float(os.environ.get("WBF_SKIP", "0.0001"))
CONF = os.environ.get("WBF_CONF", "avg")
SEED, W0, H0 = 42, 976, 1280
HDR = [
    "annotation_id",
    "image_id",
    "category_id",
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
    "score",
]


def parse_sub(p):
    """제출/예측 CSV → {image_id: [(cat, [x,y,w,h], score)]}"""
    by = defaultdict(list)
    for r in csv.DictReader(open(p)):
        by[int(r["image_id"])].append(
            (
                int(r["category_id"]),
                [
                    float(r["bbox_x"]),
                    float(r["bbox_y"]),
                    float(r["bbox_w"]),
                    float(r["bbox_h"]),
                ],
                float(r["score"]),
            )
        )
    return by


def parse_inputs():
    """WBF_INPUTS='csv:w,..' → [(dict, weight, name)]"""
    out = []
    for tok in os.environ["WBF_INPUTS"].split(","):
        tok = tok.strip()
        cp, _, w = tok.partition(":")
        out.append((parse_sub(cp), float(w) if w else 1.0, Path(cp).name))
    return out


def fuse_image(models_boxes, weights, wh):
    """한 이미지의 모델별 [(cat,[x,y,w,h],score)] 리스트 → WBF 융합 → [(cat,[x,y,w,h],score)].
    WBF는 [0,1] 정규화 xyxy 입력."""
    W, H = wh
    bl, sl, ll = [], [], []
    for boxes in models_boxes:
        b, s, l = [], [], []
        for cat, (x, y, bw, bh) in [(c, bx) for c, bx, sc in boxes]:
            b.append([x / W, y / H, (x + bw) / W, (y + bh) / H])
        for cat, bx, sc in boxes:
            s.append(sc)
            l.append(cat)
        bl.append(b)
        sl.append(s)
        ll.append(l)
    if not any(bl):
        return []
    fb, fs, fl = weighted_boxes_fusion(
        bl, sl, ll, weights=weights, iou_thr=IOU, skip_box_thr=SKIP, conf_type=CONF
    )
    out = []
    for (x1, y1, x2, y2), s, l in zip(fb, fs, fl):
        out.append((int(l), [x1 * W, y1 * H, (x2 - x1) * W, (y2 - y1) * H], float(s)))
    return out


def write_csv(path, rows_by_img):
    rows, aid = [], 1
    for iid in sorted(rows_by_img):
        for cat, (x, y, w, h), s in rows_by_img[iid]:
            rows.append(
                [
                    aid,
                    iid,
                    cat,
                    round(x, 1),
                    round(y, 1),
                    round(w, 1),
                    round(h, 1),
                    round(s, 4),
                ]
            )
            aid += 1
    with open(path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(HDR)
        wr.writerows(rows)
    return len(rows)


# ---------- fold0-val 재현 (kfold_exp와 동일) ----------
def fold0_val():
    img_anns, img_wh = defaultdict(list), {}
    for jf in paths.TRAIN_ANNOTATIONS.rglob("*.json"):
        d = json.load(open(jf))
        im = d["images"][0]
        img_wh[im["file_name"]] = (im["width"], im["height"])
        for a in d["annotations"]:
            img_anns[im["file_name"]].append(
                (int(a["category_id"]), [float(v) for v in a["bbox"]])
            )
    files = sorted(img_anns)
    sets = {fn: frozenset(c for c, _ in img_anns[fn]) for fn in files}
    combo_of = {c: i for i, c in enumerate(dict.fromkeys(sets[fn] for fn in files))}
    groups = np.array([combo_of[sets[fn]] for fn in files])
    rng = np.random.default_rng(SEED)
    uniq = np.array(sorted(set(groups)))
    rng.shuffle(uniq)
    g2f = {int(g): k for k, fl in enumerate(np.array_split(uniq, 5)) for g in fl}
    val = [fn for i, fn in enumerate(files) if g2f[int(groups[i])] == 0]
    return val, img_anns, img_wh


# ============================================================ SUBMIT
if MODE == "submit":
    from PIL import Image

    inputs = parse_inputs()
    weights = [w for _, w, _ in inputs]
    dims = {}
    for p in paths.TEST_IMAGES.glob("*.png"):
        with Image.open(p) as im:
            dims[int(p.stem)] = im.size  # (W,H)
    all_ids = sorted(set().union(*[set(d) for d, _, _ in inputs]))
    fused = {}
    for iid in all_ids:
        mb = [d.get(iid, []) for d, _, _ in inputs]
        fused[iid] = fuse_image(mb, weights, dims.get(iid, (W0, H0)))
    out = Path(
        os.environ.get("WBF_OUT", paths.RUNS / "final/submission_wbf_ensemble.csv")
    )
    n = write_csv(out, fused)
    print(f">>> 앙상블 제출 CSV: {out}", flush=True)
    print(
        f"    모델 {[(nm, w) for _, w, nm in inputs]} | IoU {IOU} skip {SKIP} conf {CONF}",
        flush=True,
    )
    print(
        f"    rows {n} · images {len(fused)}/842 · 평균 {n / max(len(fused), 1):.1f}/img",
        flush=True,
    )

# ============================================================ FOLDPRED (YOLO)
elif MODE == "foldpred":
    from ultralytics import YOLO

    val, _, _ = fold0_val()
    nid = {fn: i + 1 for i, fn in enumerate(sorted(val))}
    cm = json.load(open(paths.SSOT / "class_map.json"))
    m2c = {int(k): v for k, v in cm["model_index_to_category_id"].items()}
    model = YOLO(os.environ["WBF_MODEL"])
    fused = defaultdict(list)
    for fn in sorted(val):
        r = model.predict(
            str(paths.TRAIN_IMAGES / fn),
            conf=0.001,
            iou=0.6,
            max_det=100,
            imgsz=640,
            device="mps",
            verbose=False,
        )[0]
        for (x1, y1, x2, y2), c, s in zip(
            r.boxes.xyxy.cpu().numpy(),
            r.boxes.cls.cpu().numpy(),
            r.boxes.conf.cpu().numpy(),
        ):
            fused[nid[fn]].append(
                (
                    m2c[int(c)],
                    [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    float(s),
                )
            )
    out = Path(os.environ["WBF_OUT"])
    n = write_csv(out, fused)
    print(
        f">>> fold0 예측 CSV: {out}  (rows {n}, images {len(fused)}/{len(val)})",
        flush=True,
    )

# ============================================================ EVAL (fold0 채점)
elif MODE == "eval":
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    val, img_anns, img_wh = fold0_val()
    vals = sorted(val)
    nid = {fn: i + 1 for i, fn in enumerate(vals)}
    dim_by_id = {nid[fn]: img_wh[fn] for fn in vals}
    cats = sorted({c for fn in vals for c, _ in img_anns[fn]})
    gt = {
        "images": [
            {
                "id": nid[fn],
                "file_name": fn,
                "width": img_wh[fn][0],
                "height": img_wh[fn][1],
            }
            for fn in vals
        ],
        "annotations": [],
        "categories": [
            {"id": c}
            for c in sorted(
                set(
                    json.load(open(paths.SSOT / "class_map.json"))[
                        "model_index_to_category_id"
                    ].values()
                )
            )
        ],
    }
    aid = 1
    for fn in vals:
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
    gp = paths.RUNS / "_wbf_gt.json"
    json.dump(gt, open(gp, "w"))
    cocoGt = COCO(str(gp))
    inputs = parse_inputs()

    def score(weights, iou):
        global IOU
        IOU = iou
        dts = []
        for iid in [nid[fn] for fn in vals]:
            mb = [d.get(iid, []) for d, _, _ in inputs]
            for cat, (x, y, w, h), s in fuse_image(mb, weights, dim_by_id[iid]):
                dts.append(
                    {
                        "image_id": iid,
                        "category_id": cat,
                        "bbox": [x, y, w, h],
                        "score": s,
                    }
                )
        if not dts:
            return 0.0
        e = COCOeval(cocoGt, cocoGt.loadRes(dts), "bbox")
        e.params.iouThrs = np.linspace(0.75, 0.95, 5)
        e.evaluate()
        e.accumulate()
        e.summarize()
        return float(e.stats[0])

    base_w = [w for _, w, _ in inputs]
    if os.environ.get("WBF_GRID"):
        N = len(inputs)
        cands = [base_w, [1] * N] + [
            [2 if i == j else 1 for i in range(N)] for j in range(N)
        ]
        best = (None, None, -1)
        for iou in [0.5, 0.55, 0.6, 0.65]:
            for w in cands:
                m = score(w, iou)
                print(f"  w={w} iou={iou} → mAP@[.75:.95]={m:.4f}", flush=True)
                if m > best[2]:
                    best = (w, iou, m)
        print(
            f"\n>>> 최적: weights={best[0]} iou={best[1]} → {best[2]:.4f}", flush=True
        )
    else:
        m = score(base_w, IOU)
        print(
            f"\n>>> WBF fold0 mAP@[0.75:0.95] = {m:.4f}  (weights={base_w} iou={IOU})",
            flush=True,
        )
    print(f"    입력: {[nm for _, _, nm in inputs]}", flush=True)

else:
    sys.exit(f"WBF_MODE 미지원: {MODE} (submit/foldpred/eval)")
