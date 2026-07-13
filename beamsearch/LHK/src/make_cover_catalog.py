"""118-class 커버리지 카탈로그(그리드): 클래스별 대표 알약 크롭 + category_id + K-code + 제품명.
이름 = 우리 스키마(56) ∪ AI Hub dl_name(116). 대표이미지 = cover116(116) + real base(2 real-only). 텍스트=PIL."""

import json
import os
import sys
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths

OUT = paths.LHK / "label_audit/class_catalog_cover118.png"
FONT = "/System/Library/Fonts/AppleSDGothicNeo.ttc"


def f(sz):
    return ImageFont.truetype(FONT, sz)


BG = (24, 24, 28)
WHITE, GREY, ACC, KCOL = (
    (238, 238, 238),
    (165, 165, 170),
    (255, 210, 90),
    (120, 210, 255),
)

cmap = json.load(open(paths.SSOT / "class_map_cover.json"))
c2m = {int(k): v for k, v in cmap["category_id_to_model_index"].items()}
# 이름: 스키마(56) ∪ aihub dl_name(116)
schema = {
    c["id"]: c
    for c in json.load(
        open(paths.LHK / "handoff_realcopy/target_categories_schema.json")
    )
}
ainames = {
    int(k): v for k, v in json.load(open(paths.SSOT / "aihub_drug_names.json")).items()
}


def name_of(c):
    if c in schema and schema[c].get("name"):
        return schema[c]["name"]
    return ainames.get(c, "?")


# ---------- 대표 인스턴스(중앙값 면적=전형적 단일알약, 오버사이즈 오류박스 회피) ----------
cand = defaultdict(list)  # cat -> [(area, img_path, [x,y,w,h])]


def consider(c, area, p, box):
    cand[c].append((area, p, box))


cov = json.load(
    open(paths.PROCESSED / "kaggle_aihub_cover116/coco/annotations_coco.json")
)
covimg = {im["id"]: im["file_name"] for im in cov["images"]}
covdir = paths.PROCESSED / "kaggle_aihub_cover116/coco/images"
for a in cov["annotations"]:
    x, y, w, h = a["bbox"]
    consider(int(a["category_id"]), w * h, covdir / covimg[a["image_id"]], [x, y, w, h])
for jf in paths.TRAIN_ANNOTATIONS.rglob("*.json"):
    d = json.load(open(jf))
    fn = d["images"][0]["file_name"]
    for a in d["annotations"]:
        x, y, w, h = a["bbox"]
        consider(int(a["category_id"]), w * h, paths.TRAIN_IMAGES / fn, [x, y, w, h])

best = {}  # cat -> (area, path, box) : 면적 중앙값 인스턴스
for c, lst in cand.items():
    lst.sort(key=lambda t: t[0])
    best[c] = lst[len(lst) // 2]
miss = [c for c in c2m if c not in best]
print(
    f"대표이미지 확보: {sum(1 for c in c2m if c in best)}/{len(c2m)}  (누락 {miss})",
    flush=True,
)


def crop_square(p, box, sz=230, pad=0.18):
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
    if ch == 0 or cw == 0:
        return np.full((sz, sz, 3), 40, np.uint8)
    s = sz / max(ch, cw)
    c = cv2.resize(c, (int(cw * s), int(ch * s)), interpolation=cv2.INTER_AREA)
    canvas = np.full((sz, sz, 3), 40, np.uint8)
    oy, ox = (sz - c.shape[0]) // 2, (sz - c.shape[1]) // 2
    canvas[oy : oy + c.shape[0], ox : ox + c.shape[1]] = c
    return canvas


# ---------- 타일 ----------
TH, CAP = 230, 96
tiles = []
for c in sorted(c2m):  # category_id 오름차순
    sq = (
        crop_square(best[c][1], best[c][2])
        if c in best
        else np.full((TH, TH, 3), 40, np.uint8)
    )
    im = Image.fromarray(cv2.cvtColor(sq, cv2.COLOR_BGR2RGB))
    tile = Image.new("RGB", (TH, TH + CAP), BG)
    tile.paste(im, (0, 0))
    dr = ImageDraw.Draw(tile)
    dr.text((8, TH + 5), f"ID {c}", font=f(16), fill=ACC)
    dr.text((92, TH + 6), f"K-{c:06d}", font=f(14), fill=KCOL)
    dr.text((TH - 44, TH + 6), f"idx{c2m[c]}", font=f(12), fill=GREY)
    # 이름 2줄 래핑
    nm, line, yy = name_of(c), "", TH + 30
    for ch in nm:
        if dr.textlength(line + ch, font=f(14)) > TH - 12:
            dr.text((8, yy), line, font=f(14), fill=WHITE)
            yy += 19
            line = ch
            if yy > TH + 68:
                line += "…"
                break
        else:
            line += ch
    dr.text((8, yy), line, font=f(14), fill=WHITE)
    tiles.append(tile)

# ---------- 그리드 ----------
COLS, PAD = 8, 8
maxh = TH + CAP
rowsN = (len(tiles) + COLS - 1) // COLS
GW = COLS * TH + (COLS + 1) * PAD
HDR = 120
canvas = Image.new("RGB", (GW, HDR + rowsN * (maxh + PAD) + PAD), (16, 16, 18))
for i, t in enumerate(tiles):
    rr, cc = divmod(i, COLS)
    canvas.paste(t, (PAD + cc * (TH + PAD), HDR + PAD + rr * (maxh + PAD)))

d = ImageDraw.Draw(canvas)
d.text(
    (18, 16),
    "경구약제 커버리지 클래스 카탈로그 — 118종 (새 [AI12] 대회 대응)",
    font=f(30),
    fill=WHITE,
)
d.text(
    (20, 60),
    "ID(category_id) · K-code · 제품명  |  이름 = 대회 스키마(56) ∪ AI Hub dl_name(116)",
    font=f(17),
    fill=(195, 195, 200),
)
d.text(
    (20, 88),
    "대표 이미지 = cover116(AI Hub 조합) + real(대회기본, ID 3351·3483)  |  학습 클래스 = 이 118종 전체",
    font=f(15),
    fill=(150, 150, 155),
)
canvas.save(OUT)
print(f">>> 저장: {OUT}  ({canvas.width}x{canvas.height}, {len(tiles)}종)", flush=True)
