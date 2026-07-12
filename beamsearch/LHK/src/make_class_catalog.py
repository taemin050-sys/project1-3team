"""56클래스 카탈로그(그리드): 클래스별 대표 알약 크롭 + category_id + 제품명(+모양/색). 텍스트=PIL 한글.
대표 크롭 = real 학습데이터에서 해당 클래스 최대면적 인스턴스(가장 선명). 없으면 aihub에서 보완."""

import json
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths

OUT = paths.LHK / "label_audit/class_catalog_56.png"
FONT = "/System/Library/Fonts/AppleSDGothicNeo.ttc"


def f(sz):
    return ImageFont.truetype(FONT, sz)


BG = (24, 24, 28)
WHITE, GREY, ACC = (238, 238, 238), (165, 165, 170), (255, 210, 90)

cm = json.load(open(paths.SSOT / "class_map.json"))
c2m = {int(k): v for k, v in cm["category_id_to_model_index"].items()}
schema = {
    c["id"]: c
    for c in json.load(
        open(paths.LHK / "handoff_realcopy/target_categories_schema.json")
    )
}

# ---------- 클래스별 대표 인스턴스(최대면적) ----------
best = {}  # cat_id -> (area, img_path, [x,y,w,h])
for jf in paths.TRAIN_ANNOTATIONS.rglob("*.json"):
    d = json.load(open(jf))
    fn = d["images"][0]["file_name"]
    p = paths.TRAIN_IMAGES / fn
    for a in d["annotations"]:
        c = int(a["category_id"])
        x, y, w, h = a["bbox"]
        area = w * h
        if c not in best or area > best[c][0]:
            best[c] = (area, p, [float(v) for v in a["bbox"]])
# aihub 보완(real에 없는 클래스)
miss = [c for c in c2m if c not in best]
if miss:
    ac = json.load(
        open(paths.PROCESSED / "kaggle_aihub_full_inpaint/coco/annotations_coco.json")
    )
    id2fn = {im["id"]: im["file_name"] for im in ac["images"]}
    adir = paths.PROCESSED / "kaggle_aihub_full_inpaint/coco/images"
    for a in ac["annotations"]:
        c = int(a["category_id"])
        if c in miss:
            x, y, w, h = a["bbox"]
            if c not in best or w * h > best[c][0]:
                best[c] = (
                    w * h,
                    adir / id2fn[a["image_id"]],
                    [float(v) for v in a["bbox"]],
                )

print(
    f"대표이미지 확보: {sum(1 for c in c2m if c in best)}/{len(c2m)} 클래스", flush=True
)


def crop_square(p, box, sz=240, pad=0.18):
    img = cv2.imread(str(p))
    if img is None:
        return np.full((sz, sz, 3), 40, np.uint8)
    H, W = img.shape[:2]
    x, y, w, h = box
    mx, my = w * pad, h * pad
    x0, y0 = int(max(x - mx, 0)), int(max(y - my, 0))
    x1, y1 = int(min(x + w + mx, W)), int(min(y + h + my, H))
    c = img[y0:y1, x0:x1]
    ch, cw = c.shape[:2]
    s = sz / max(ch, cw)
    c = cv2.resize(c, (int(cw * s), int(ch * s)), interpolation=cv2.INTER_AREA)
    canvas = np.full((sz, sz, 3), 40, np.uint8)
    oy, ox = (sz - c.shape[0]) // 2, (sz - c.shape[1]) // 2
    canvas[oy : oy + c.shape[0], ox : ox + c.shape[1]] = c
    return canvas


# ---------- 타일 ----------
TH, CAP = 240, 92
tiles = []
for c in sorted(c2m, key=lambda x: c2m[x]):  # model_index 순
    sq = (
        crop_square(best[c][1], best[c][2])
        if c in best
        else np.full((TH, TH, 3), 40, np.uint8)
    )
    im = Image.fromarray(cv2.cvtColor(sq, cv2.COLOR_BGR2RGB))
    tile = Image.new("RGB", (TH, TH + CAP), BG)
    tile.paste(im, (0, 0))
    dr = ImageDraw.Draw(tile)
    s = schema.get(c, {})
    name = s.get("name", "?")
    dr.text((8, TH + 6), f"ID {c}", font=f(17), fill=ACC)
    dr.text((70, TH + 7), f"idx {c2m[c]}", font=f(13), fill=GREY)
    # 제품명 2줄 래핑
    line, yy = "", TH + 30
    for ch in name:
        if dr.textlength(line + ch, font=f(15)) > TH - 14:
            dr.text((8, yy), line, font=f(15), fill=WHITE)
            yy += 20
            line = ch
            if yy > TH + 50:
                line += "…"
                break
        else:
            line += ch
    dr.text((8, yy), line, font=f(15), fill=WHITE)
    meta = " · ".join(
        x
        for x in [
            s.get("shape", ""),
            s.get("color", ""),
            (s.get("imprint_front") or ""),
        ]
        if x
    )
    dr.text((8, TH + CAP - 18), meta[:26], font=f(12), fill=(140, 140, 150))
    tiles.append(tile)

# ---------- 그리드 ----------
COLS, PAD = 7, 8
TWF = TH
maxh = TH + CAP
rowsN = (len(tiles) + COLS - 1) // COLS
GW = COLS * TWF + (COLS + 1) * PAD
HDR = 118
canvas = Image.new("RGB", (GW, HDR + rowsN * (maxh + PAD) + PAD), (16, 16, 18))
for i, t in enumerate(tiles):
    rr, cc = divmod(i, COLS)
    canvas.paste(t, (PAD + cc * (TWF + PAD), HDR + PAD + rr * (maxh + PAD)))

d = ImageDraw.Draw(canvas)
d.text(
    (18, 16), "경구약제 클래스 카탈로그 — 56종 (Test=40 ⊆ 56)", font=f(30), fill=WHITE
)
d.text(
    (20, 60),
    "category_id · 제품명 · 모양/색/각인  |  대표 이미지 = 학습데이터 최대 인스턴스 크롭",
    font=f(17),
    fill=(195, 195, 200),
)
d.text(
    (20, 88),
    "※ Test 40클래스는 무라벨이라 56 중 특정 불가 → 전체 56종 표시 (채점은 40에 대해서만)",
    font=f(14),
    fill=(150, 150, 155),
)

canvas.save(OUT)
print(f">>> 저장: {OUT}  ({canvas.width}x{canvas.height}, {len(tiles)}종)", flush=True)
