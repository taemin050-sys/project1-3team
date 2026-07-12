"""라벨 오류 자동탐지 (1차): 강한 모델(0.985) 예측 vs GT 불일치 + 기하 휴리스틱 → 의심 박스 랭킹 + 검수 컨택트시트.
대상: real232(corrected) + aihub7836 (synth696 제외=프로그램 라벨). '잘못 그려진 bbox'(위치/크기)는 non-OOF여도 검출됨.
출력: lab/label_audit/{suspects.csv, review/*.jpg, contact.jpg}. 사람은 상위 랭킹만 수분 검수."""

import csv
import json
import os
import sys
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths

TA, TI, PROCESSED, SSOT, RUNS = (
    paths.TRAIN_ANNOTATIONS,
    paths.TRAIN_IMAGES,
    paths.PROCESSED,
    paths.SSOT,
    paths.RUNS,
)
# 파라미터화(커버리지 감사): 모델·클래스맵·감사대상 synth·출력·디바이스 env 선택
BEST = os.environ.get("AUDIT_BEST") or str(
    RUNS / "final/s11s_full232_synth696_aihubfull/weights/best.pt"
)
AUDIT_SYNTH = os.environ.get("AUDIT_SYNTH", "kaggle_aihub_full_inpaint")
DEVICE = os.environ.get("AUDIT_DEVICE", "cpu")
OUT = paths.LAB / os.environ.get("AUDIT_OUT", "label_audit")
(OUT / "review").mkdir(parents=True, exist_ok=True)
cm = json.load(open(SSOT / os.environ.get("AUDIT_CLASSMAP", "class_map.json")))
m2c = {int(k): v for k, v in cm["model_index_to_category_id"].items()}
c2m = {int(k): v for k, v in cm["category_id_to_model_index"].items()}
sc = paths.LHK / "handoff_realcopy/target_categories_schema.json"
name = {c["id"]: c.get("name", "") for c in json.load(open(sc))} if sc.exists() else {}


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    u = aw * ah + bw * bh - inter
    return inter / u if u > 0 else 0.0


# ---------- GT 수집 (real corrected + aihub) ----------
samples = []  # (source, img_path, W, H, [(cat_id, [x,y,w,h])])
img_anns, img_wh = defaultdict(list), {}
for jf in TA.rglob("*.json"):
    d = json.load(open(jf))
    im = d["images"][0]
    img_wh[im["file_name"]] = (im["width"], im["height"])
    for a in d["annotations"]:
        img_anns[im["file_name"]].append(
            (int(a["category_id"]), [float(v) for v in a["bbox"]])
        )
corr = paths.LHK / "data/gt_corrected/corrections.json"
if corr.exists():
    for fn, adds in json.load(open(corr)).get("type_A_autofill", {}).items():
        for a in adds:
            img_anns[fn].append((int(a["category_id"]), a["bbox"]))
for fn, anns in img_anns.items():
    W, H = img_wh[fn]
    samples.append(("real", TI / fn, W, H, anns))
# aihub
ac = json.load(open(PROCESSED / AUDIT_SYNTH / "coco/annotations_coco.json"))
awh = {im["id"]: (im["width"], im["height"], im["file_name"]) for im in ac["images"]}
aby = defaultdict(list)
for a in ac["annotations"]:
    aby[a["image_id"]].append((int(a["category_id"]), [float(v) for v in a["bbox"]]))
adir = PROCESSED / AUDIT_SYNTH / "coco/images"
for iid, (W, H, fn) in awh.items():
    samples.append(("aihub", adir / fn, W, H, aby[iid]))
print(
    f"감사 대상: {len(samples)}장 (real {sum(1 for s in samples if s[0] == 'real')} + aihub {sum(1 for s in samples if s[0] == 'aihub')})",
    flush=True,
)

# 클래스별 정규면적 통계(휴리스틱용)
areas_by = defaultdict(list)
for _, _, W, H, anns in samples:
    for c, (x, y, w, h) in anns:
        areas_by[c].append(w * h / (W * H))
amed = {c: float(np.median(v)) for c, v in areas_by.items()}

# ---------- 예측 + 불일치/휴리스틱 스코어 ----------
model = YOLO(str(BEST))
rows = []
for src, p, W, H, anns in samples:
    if not Path(p).exists():
        continue
    res = model.predict(
        str(p), conf=0.25, iou=0.6, max_det=30, imgsz=640, device=DEVICE, verbose=False
    )[0]
    preds = []
    for b, c, s in zip(
        res.boxes.xyxy.cpu().numpy(),
        res.boxes.cls.cpu().numpy(),
        res.boxes.conf.cpu().numpy(),
    ):
        x1, y1, x2, y2 = b
        preds.append(
            (
                m2c[int(c)],
                [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                float(s),
            )
        )
    used = set()
    flags = []
    for gi, (gc, gb) in enumerate(anns):
        # 최적 매칭 예측
        best_j, best_iou = -1, 0
        for j, (pc, pb, ps) in enumerate(preds):
            i = iou(gb, pb)
            if i > best_iou:
                best_iou, best_j = i, j
        gx, gy, gw, gh = gb
        # 기하 휴리스틱
        if gw <= 2 or gh <= 2:
            flags.append(("degenerate", gi, 3.0))
        if gx < -1 or gy < -1 or gx + gw > W + 2 or gy + gh > H + 2:
            flags.append(("out_of_bounds", gi, 2.5))
        ar = gw / max(gh, 1)
        if ar > 6 or ar < 1 / 6:
            flags.append(("extreme_aspect", gi, 1.5))
        na = gw * gh / (W * H)
        if gc in amed and amed[gc] > 0 and (na > amed[gc] * 4 or na < amed[gc] / 4):
            flags.append(("size_outlier", gi, 1.5))
        # 모델 불일치
        if best_iou < 0.3:
            flags.append(
                ("no_pred_match(misplaced?)", gi, 3.0)
            )  # GT박스에 매칭 검출 없음=위치 오류 의심
        else:
            used.add(best_j)
            pc = preds[best_j][0]
            if pc != gc:
                flags.append((f"class_mismatch(pred {pc})", gi, 2.5))
            elif best_iou < 0.6:
                flags.append(("loose_bbox", gi, 1.0))
    # 고신뢰 예측인데 GT 없음 = 누락 라벨 의심
    for j, (pc, pb, ps) in enumerate(preds):
        if j not in used and ps > 0.6:
            flags.append((f"missing_gt(pred {pc} @{ps:.2f})", -1, 2.0 * ps))
    if flags:
        score = sum(w for _, _, w in flags)
        rows.append(
            {
                "source": src,
                "img": str(p),
                "W": W,
                "H": H,
                "score": round(score, 2),
                "n_gt": len(anns),
                "flags": flags,
                "anns": anns,
                "preds": preds,
            }
        )

rows.sort(key=lambda r: -r["score"])
print(f"의심 이미지: {len(rows)}장 (상위 랭킹부터)", flush=True)

# CSV
with open(OUT / "suspects.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["rank", "source", "img", "score", "n_gt", "flag_types"])
    for i, r in enumerate(rows, 1):
        w.writerow(
            [
                i,
                r["source"],
                Path(r["img"]).name,
                r["score"],
                r["n_gt"],
                ";".join(sorted(set(t for t, _, _ in r["flags"]))),
            ]
        )

# 상위 60 컨택트시트 (초록=정상매칭 GT, 빨강=의심 GT, 파랑=모델예측)
TOPN = 60
tiles = []
for r in rows[:TOPN]:
    img = cv2.imread(r["img"])
    if img is None:
        continue
    susp = {gi for _, gi, _ in r["flags"] if gi >= 0}
    for gi, (gc, (x, y, w, h)) in enumerate(r["anns"]):
        col = (0, 0, 255) if gi in susp else (0, 200, 0)
        cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), col, 4)
    for pc, (x, y, w, h), ps in r["preds"]:
        cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (255, 150, 0), 2)
    cv2.putText(
        img,
        f"{r['score']:.1f} {r['source']}",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (0, 0, 255),
        3,
    )
    cv2.imwrite(
        str(OUT / "review" / f"{r['score']:07.2f}_{Path(r['img']).stem}.jpg"), img
    )
    tiles.append(cv2.resize(img, (300, 394)))
if tiles:
    cols = 6
    grid = [tiles[i : i + cols] for i in range(0, len(tiles), cols)]
    grid[-1] += [np.full((394, 300, 3), 30, np.uint8)] * (cols - len(grid[-1]))
    cv2.imwrite(str(OUT / "contact.jpg"), np.vstack([np.hstack(r) for r in grid]))
print(
    f">>> 산출: {OUT}/suspects.csv (랭킹), review/ (개별), contact.jpg (상위{TOPN})",
    flush=True,
)
print(
    f">>> 상위 10 의심: {[(Path(r['img']).name[:24], r['score']) for r in rows[:10]]}",
    flush=True,
)
