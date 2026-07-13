"""오류분석(clean fold0 모델): fold0-val 예측 → GT매칭 → 오류 분류(오분류/미검출/오검출) +
클래스별 AP@[0.75:0.95] + 혼동쌍 + 오류 컨택트시트. 표적튜닝 안내용.
※ fold0-val=51장/~166박스로 희소 → 절대수치보다 '패턴' 신호로 해석."""

import json
import os
import sys
from collections import defaultdict, Counter

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths

BEST = paths.RUNS / "kfold/f0_clean_all_11s/weights/best.pt"
OUT = paths.LHK / "label_audit"
FONT = "/System/Library/Fonts/AppleSDGothicNeo.ttc"
IMGSZ, SEED, W, H = 640, 42, 976, 1280
cm = json.load(open(paths.SSOT / "class_map.json"))
c2m = {int(k): v for k, v in cm["category_id_to_model_index"].items()}
m2c = {int(k): v for k, v in cm["model_index_to_category_id"].items()}
schema = {
    c["id"]: c.get("name", "")
    for c in json.load(
        open(paths.LHK / "handoff_realcopy/target_categories_schema.json")
    )
}


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    return inter / (aw * ah + bw * bh - inter + 1e-9)


# ---------- fold0-val 재현 (kfold_exp와 동일 GroupKFold) ----------
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
print(f"fold0-val: {len(val)}장 / {sum(len(img_anns[f]) for f in val)}박스", flush=True)

# ---------- 예측 ----------
model = YOLO(str(BEST))
preds = {}
for fn in val:
    r = model.predict(
        str(paths.TRAIN_IMAGES / fn),
        conf=0.25,
        iou=0.6,
        max_det=30,
        imgsz=IMGSZ,
        device="mps",
        verbose=False,
    )[0]
    preds[fn] = [
        (m2c[int(c)], [float(x1), float(y1), float(x2 - x1), float(y2 - y1)], float(s))
        for (x1, y1, x2, y2), c, s in zip(
            r.boxes.xyxy.cpu().numpy(),
            r.boxes.cls.cpu().numpy(),
            r.boxes.conf.cpu().numpy(),
        )
    ]

# ---------- 매칭 & 오류 분류 (IoU 0.5) ----------
n_tp = n_mis = n_fn = n_fp = 0
confuse = Counter()  # (gt_cls, pred_cls) 오분류
per_cls = defaultdict(lambda: [0, 0, 0])  # cls -> [n_gt, hit, miss]
errors = []  # (fn, kind, gt_c, pred_c, box)
for fn in val:
    gts = img_anns[fn]
    pd = sorted(preds[fn], key=lambda z: -z[2])
    used = set()
    for gc, gb in gts:
        per_cls[gc][0] += 1
        j, best = -1, 0.5
        for k, (pc, pb, ps) in enumerate(pd):
            if k in used:
                continue
            i = iou(gb, pb)
            if i >= best:
                best, j = i, k
        if j < 0:
            n_fn += 1
            per_cls[gc][2] += 1
            errors.append((fn, "미검출(FN)", gc, None, gb))
        else:
            used.add(j)
            pc = pd[j][0]
            if pc == gc:
                n_tp += 1
                per_cls[gc][1] += 1
            else:
                n_mis += 1
                confuse[(gc, pc)] += 1
                errors.append((fn, "오분류", gc, pc, gb))
    for k, (pc, pb, ps) in enumerate(pd):
        if k not in used and ps > 0.5:
            n_fp += 1
            errors.append((fn, "오검출(FP)", None, pc, pb))

# ---------- 클래스별 AP@[0.75:0.95] (cocoeval) ----------
nid = {fn: i + 1 for i, fn in enumerate(val)}
gtc = {
    "images": [{"id": nid[f], "file_name": f, "width": W, "height": H} for f in val],
    "annotations": [],
    "categories": [{"id": c} for c in sorted(set(m2c.values()))],
}
aid = 1
for fn in val:
    for c, (x, y, w, h) in img_anns[fn]:
        gtc["annotations"].append(
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
gp = OUT / "_ea_gt.json"
json.dump(gtc, open(gp, "w"))
dts = []
for fn in val:
    r = model.predict(
        str(paths.TRAIN_IMAGES / fn),
        conf=0.001,
        iou=0.6,
        max_det=100,
        imgsz=IMGSZ,
        device="mps",
        verbose=False,
    )[0]
    for (x1, y1, x2, y2), c, s in zip(
        r.boxes.xyxy.cpu().numpy(),
        r.boxes.cls.cpu().numpy(),
        r.boxes.conf.cpu().numpy(),
    ):
        dts.append(
            {
                "image_id": nid[fn],
                "category_id": m2c[int(c)],
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score": float(s),
            }
        )
coco = COCO(str(gp))
e = COCOeval(coco, coco.loadRes(dts), "bbox")
e.params.iouThrs = np.linspace(0.75, 0.95, 5)
e.evaluate()
e.accumulate()
prec = e.eval["precision"]  # [T,R,K,A,M]
cats = [c["id"] for c in coco.dataset["categories"]]
ap_cls = {}
for ki, c in enumerate(cats):
    pr = prec[:, :, ki, 0, -1]
    pr = pr[pr > -1]
    if per_cls[c][0] > 0:
        ap_cls[c] = float(pr.mean()) if pr.size else 0.0

# ---------- 리포트 ----------
print("\n=== 오류 요약 (fold0-val, conf0.25/IoU0.5) ===", flush=True)
print(
    f"  정답(TP) {n_tp} · 오분류 {n_mis} · 미검출(FN) {n_fn} · 오검출(FP) {n_fp}",
    flush=True,
)
print("\n=== 혼동쌍 (GT→예측, 상위) ===", flush=True)
for (g, p), n in confuse.most_common(10):
    print(
        f"  {g}({schema.get(g, '')[:12]}) → {p}({schema.get(p, '')[:12]}) : {n}회",
        flush=True,
    )
print("\n=== 취약 클래스 (fold0-val 등장분, AP@[.75:.95] 오름차순) ===", flush=True)
weak = sorted(ap_cls.items(), key=lambda kv: kv[1])[:12]
for c, ap in weak:
    ng, hit, miss = per_cls[c]
    print(
        f"  AP {ap:.3f}  id {c} {schema.get(c, '')[:16]:16s} (GT {ng}, 적중 {hit}, 미검출 {miss})",
        flush=True,
    )


# ---------- 오류 컨택트시트 ----------
def fnt(s):
    return ImageFont.truetype(FONT, s)


tiles = []
for fn, kind, gc, pc, box in errors[:48]:
    im = Image.open(paths.TRAIN_IMAGES / fn).convert("RGB")
    dr = ImageDraw.Draw(im)
    x, y, w, h = box
    col = (
        (240, 80, 80)
        if kind.startswith("미검출")
        else (80, 160, 255)
        if kind.startswith("오검출")
        else (255, 180, 60)
    )
    dr.rectangle([x, y, x + w, y + h], outline=col, width=6)
    TWt = 330
    th = int(TWt * im.height / im.width)
    tile = im.resize((TWt, th), Image.LANCZOS)
    cap = Image.new("RGB", (TWt, 66), (24, 24, 28))
    cd = ImageDraw.Draw(cap)
    cd.text((7, 5), kind, font=fnt(16), fill=col)
    txt = (
        f"GT {gc}"
        + (f" → 예측 {pc}" if pc is not None and gc is not None else "")
        + (f"예측 {pc}" if gc is None else "")
    )
    cd.text((7, 28), txt, font=fnt(15), fill=(220, 220, 220))
    nm = schema.get(gc if gc is not None else pc, "")[:20]
    cd.text((7, 47), nm, font=fnt(13), fill=(150, 150, 155))
    full = Image.new("RGB", (TWt, th + 66), (24, 24, 28))
    full.paste(tile, (0, 0))
    full.paste(cap, (0, th))
    tiles.append(full)
if tiles:
    COLS = 6
    maxh = max(t.height for t in tiles)
    PADp = 7
    rN = (len(tiles) + COLS - 1) // COLS
    GW = COLS * 330 + (COLS + 1) * PADp
    cv = Image.new("RGB", (GW, 70 + rN * (maxh + PADp) + PADp), (16, 16, 18))
    for i, t in enumerate(tiles):
        rr, cc = divmod(i, COLS)
        cv.paste(t, (PADp + cc * (330 + PADp), 70 + PADp + rr * (maxh + PADp)))
    dd = ImageDraw.Draw(cv)
    dd.text(
        (16, 16),
        f"오류 케이스 — clean fold0 (오분류 주황 · 미검출 빨강 · 오검출 파랑)  총 {len(errors)}건",
        font=fnt(22),
        fill=(238, 238, 240),
    )
    cv.save(OUT / "error_cases.png")
    print(
        f"\n>>> 오류 컨택트시트: {OUT}/error_cases.png ({len(tiles)}/{len(errors)}건)",
        flush=True,
    )
