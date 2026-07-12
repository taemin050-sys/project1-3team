"""스크럼 공유용: 라벨 자동정리 오류 의심 이미지 그리드 컨택트시트.
각 GT 박스에 category_id 라벨, 타일별 캡션(순위/소스/점수/클래스/사유), 상단 요약·범례. → PNG.
텍스트는 PIL(한글 폰트)로 렌더링."""

import csv
import json
import os
import sys
from pathlib import Path
from collections import defaultdict

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths

LA = paths.LHK / "label_audit"
REVIEW = LA / "review"
OUT = LA / "scrum_suspects_grid.png"
FONT = "/System/Library/Fonts/AppleSDGothicNeo.ttc"


def f(sz):
    return ImageFont.truetype(FONT, sz)


BG = (22, 22, 26)
RED, GREEN, BLUE, WHITE, GREY = (
    (240, 90, 90),
    (110, 220, 130),
    (70, 170, 240),
    (235, 235, 235),
    (170, 170, 170),
)

# ---------- GT 조회 (파일명 → [(cat_id,[x,y,w,h])]) ----------
gt = defaultdict(list)
ac = json.load(
    open(paths.PROCESSED / "kaggle_aihub_full_inpaint/coco/annotations_coco.json")
)
id2fn = {im["id"]: im["file_name"] for im in ac["images"]}
for a in ac["annotations"]:
    gt[id2fn[a["image_id"]]].append(
        (int(a["category_id"]), [float(v) for v in a["bbox"]])
    )
for jf in paths.TRAIN_ANNOTATIONS.rglob("*.json"):
    d = json.load(open(jf))
    fn = d["images"][0]["file_name"]
    gt[fn] = [
        (int(a["category_id"]), [float(v) for v in a["bbox"]]) for a in d["annotations"]
    ]

rows = list(csv.DictReader(open(LA / "suspects.csv")))[:60]

FLAGMAP = [
    ("missing_gt", "라벨누락"),
    ("no_pred_match", "위치오류"),
    ("class_mismatch", "클래스오류"),
    ("loose_bbox", "박스느슨"),
    ("out_of_bounds", "경계이탈"),
    ("size_outlier", "크기이상"),
    ("extreme_aspect", "비율이상"),
    ("degenerate", "퇴화박스"),
]


def short_flag(ft):
    s = []
    for key, kor in FLAGMAP:
        if any(t.startswith(key) for t in ft.split(";")) and kor not in s:
            s.append(kor)
    return "·".join(s)


# ---------- 타일 ----------
TW, CAP = 330, 82
tiles = []
for r in rows:
    stem = Path(r["img"]).stem
    rp = next(REVIEW.glob(f"*{stem}.jpg"), None)
    if not rp:
        continue
    im = Image.open(rp).convert("RGB")
    dr = ImageDraw.Draw(im)
    bf = f(max(16, im.width // 45))
    cats = []
    for c, (x, y, bw, bh) in gt.get(r["img"], []):
        cats.append(c)
        tx, ty = int(max(x, 1)), int(max(y, 1))
        lab = str(c)
        l, t, rr, b = dr.textbbox((tx, ty), lab, font=bf)
        dr.rectangle([l - 2, t - 2, rr + 2, b + 2], fill=(0, 0, 0))
        dr.text((tx, ty), lab, font=bf, fill=(255, 240, 120))
    th = int(TW * im.height / im.width)
    tile = im.resize((TW, th), Image.LANCZOS)
    # 캡션
    cap = Image.new("RGB", (TW, CAP), BG)
    cd = ImageDraw.Draw(cap)
    scol = RED if r["source"] == "aihub" else BLUE
    cd.text(
        (7, 5),
        f"#{r['rank']}  {r['source']}  score {r['score']}",
        font=f(16),
        fill=scol,
    )
    cls = ",".join(str(c) for c in sorted(set(cats)))
    cd.text(
        (7, 30),
        f"cls: {cls if len(cls) <= 40 else cls[:38] + '..'}",
        font=f(15),
        fill=GREY,
    )
    cd.text(
        (7, 55),
        f"사유: {short_flag(r['flag_types'])}",
        font=f(15),
        fill=(120, 220, 255),
    )
    full = Image.new("RGB", (TW, th + CAP), BG)
    full.paste(tile, (0, 0))
    full.paste(cap, (0, th))
    tiles.append(full)

# ---------- 그리드 ----------
COLS, PAD = 6, 7
maxh = max(t.height for t in tiles)
rowsN = (len(tiles) + COLS - 1) // COLS
GW = COLS * TW + (COLS + 1) * PAD
HDR = 158
canvas = Image.new("RGB", (GW, HDR + rowsN * (maxh + PAD) + PAD), (16, 16, 18))
for i, t in enumerate(tiles):
    rr, cc = divmod(i, COLS)
    x = PAD + cc * (TW + PAD)
    y = HDR + PAD + rr * (maxh + PAD)
    canvas.paste(t, (x, y))

# ---------- 헤더 ----------
d = ImageDraw.Draw(canvas)
n_ai = sum(1 for r in rows if r["source"] == "aihub")
d.text((18, 16), "라벨 자동정리 — 오류 의심 이미지", font=f(30), fill=WHITE)
d.text(
    (20, 62),
    f"상위 {len(rows)}장 표시  ·  전체 246 / 8068 (3.0%)  ·  aihub 240 · real 6",
    font=f(18),
    fill=(195, 195, 200),
)
lg = [
    ("빨강 = 의심 GT박스", RED),
    ("초록 = 정상매칭 GT", GREEN),
    ("파랑 = 모델검출", BLUE),
    ("박스 위 숫자 = category_id", (255, 240, 120)),
]
x = 20
for txt, col in lg:
    d.rectangle([x, 96, x + 20, 114], fill=col)
    d.text((x + 26, 95), txt, font=f(16), fill=(215, 215, 220))
    x += 26 + int(d.textlength(txt, font=f(16))) + 30
d.text(
    (20, 128),
    "캡션: 순위/소스/점수  ·  cls=category_id 목록  ·  사유=자동감사 플래그",
    font=f(15),
    fill=(140, 140, 145),
)

canvas.save(OUT)
print(
    f">>> 저장: {OUT}  ({canvas.width}x{canvas.height}, 타일 {len(tiles)})", flush=True
)
