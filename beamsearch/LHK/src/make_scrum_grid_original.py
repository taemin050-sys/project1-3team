"""스크럼 공유용(원본 버전): AI Hub 인페인트 '전' 원본 이미지에 초록(우리클래스)/빨강(비우리클래스)
박스 + category_id를 얹은 의심 이미지 그리드. aihub 의심만 대상(real은 원본=대회기본). 텍스트=PIL 한글."""

import csv
import io
import json
import os
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths

LA = paths.LHK / "label_audit"
OUT = LA / "scrum_suspects_grid_original.png"
FONT = "/System/Library/Fonts/AppleSDGothicNeo.ttc"
ALLOW = Path("/Volumes/USB 1T/사용 허가 조합/01.데이터/1.Training")
TSD, TLD = (
    ALLOW / "원천데이터/경구약제조합 5000종",
    ALLOW / "라벨링데이터/경구약제조합 5000종",
)


def f(sz):
    return ImageFont.truetype(FONT, sz)


BG = (22, 22, 26)
GREEN, RED, WHITE, GREY = (70, 200, 90), (240, 80, 80), (235, 235, 235), (170, 170, 170)

cm = json.load(open(paths.SSOT / "class_map.json"))
our = set(int(k) for k in cm["category_id_to_model_index"])

# ---------- 대상: 상위 aihub 의심 60 ----------
rows = [r for r in csv.DictReader(open(LA / "suspects.csv")) if r["source"] == "aihub"][
    :60
]
need = {r["img"].split("aihub_")[-1][:-4].split("_")[0] for r in rows}  # 필요한 combo만

# ---------- zip 인덱스(필요 combo만) ----------
ts_idx, tl_idx, zcache = {}, {}, {}
for z in ["TS_1", "TS_3", "TS_4", "TS_5", "TS_6", "TS_7", "TS_8"]:
    zf = zipfile.ZipFile(TSD / f"{z}_조합.zip")
    zcache[z] = zf
    for n in zf.namelist():
        if n.endswith(".png") and "_index" not in n and "/" in n:
            combo = n.split("/")[0]
            if combo in need:
                ts_idx.setdefault(combo, (z, []))[1].append(n)
for z in ["TL_1", "TL_3", "TL_4", "TL_5", "TL_6", "TL_7", "TL_8"]:
    zf = zipfile.ZipFile(TLD / f"{z}_조합.zip")
    zcache["L" + z] = zf
    for n in zf.namelist():
        if "_json/" in n:
            combo = n.split("_json/")[0]
            if combo in need and combo not in tl_idx:
                tl_idx[combo] = zf


def boxes_for(combo, base):
    zf = tl_idx.get(combo)
    if not zf:
        return []
    bs = []
    for d in combo.split("-")[1:]:
        try:
            jd = json.load(io.BytesIO(zf.read(f"{combo}_json/K-{d}/{base}.json")))
        except KeyError:
            continue
        a = jd.get("annotations", [])
        if a and len(a[0].get("bbox", [])) == 4:
            bs.append((int(d), [float(v) for v in a[0]["bbox"]]))
    return bs


# ---------- 타일 ----------
TW, CAP = 330, 82
tiles = []
for r in rows:
    stem = r["img"].split("aihub_")[-1][:-4]
    combo = stem.split("_")[0]
    z_m = ts_idx.get(combo)
    if not z_m:
        continue
    z, members = z_m
    mem = next((m for m in members if Path(m).stem == stem), None)
    if not mem:
        continue
    bgr = cv2.imdecode(np.frombuffer(zcache[z].read(mem), np.uint8), cv2.IMREAD_COLOR)
    im = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    dr = ImageDraw.Draw(im)
    bf = f(max(18, im.width // 42))
    n_our = n_non = 0
    for c, (x, y, bw, bh) in boxes_for(combo, stem):
        col = GREEN if c in our else RED
        n_our += c in our
        n_non += c not in our
        dr.rectangle([x, y, x + bw, y + bh], outline=col, width=5)
        tx, ty = int(max(x, 1)), int(max(y, 1))
        lab = str(c)
        l, t, rr, b = dr.textbbox((tx, ty), lab, font=bf)
        dr.rectangle([l - 2, t - 2, rr + 2, b + 2], fill=(0, 0, 0))
        dr.text((tx, ty), lab, font=bf, fill=col)
    th = int(TW * im.height / im.width)
    tile = im.resize((TW, th), Image.LANCZOS)
    cap = Image.new("RGB", (TW, CAP), BG)
    cd = ImageDraw.Draw(cap)
    cd.text(
        (7, 5),
        f"#{r['rank']}  aihub  score {r['score']}",
        font=f(16),
        fill=(255, 200, 90),
    )
    cd.text(
        (7, 30),
        f"우리클래스 {n_our}개(초록) · 비우리 {n_non}개(빨강)",
        font=f(15),
        fill=GREY,
    )
    ourc = ",".join(str(c) for c, (x, y, w, h) in boxes_for(combo, stem) if c in our)
    cd.text(
        (7, 55),
        f"우리 cls: {ourc if len(ourc) <= 38 else ourc[:36] + '..'}",
        font=f(15),
        fill=(150, 230, 160),
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
    canvas.paste(t, (PAD + cc * (TW + PAD), HDR + PAD + rr * (maxh + PAD)))

d = ImageDraw.Draw(canvas)
d.text(
    (18, 16),
    "라벨 자동정리 — 오류 의심 이미지 (AI Hub 원본, 인페인트 전)",
    font=f(28),
    fill=WHITE,
)
d.text(
    (20, 60),
    f"aihub 의심 상위 {len(tiles)}장  ·  원본 라벨 그대로 표시  ·  전체 246 / 8068 (3.0%)",
    font=f(18),
    fill=(195, 195, 200),
)
lg = [
    ("초록 = 우리 56클래스 (보존)", GREEN),
    ("빨강 = 비-우리 클래스 (인페인트 제거)", RED),
    ("박스 위 숫자 = category_id", (255, 240, 120)),
]
x = 20
for txt, col in lg:
    d.rectangle([x, 96, x + 20, 114], fill=col)
    d.text((x + 26, 95), txt, font=f(16), fill=(215, 215, 220))
    x += 26 + int(d.textlength(txt, font=f(16))) + 30
d.text(
    (20, 128),
    "요점: 우리가 '보존'하는 초록 박스 중 어긋난 것이 오류의 정체 (소스 라벨 오류)",
    font=f(15),
    fill=(140, 140, 145),
)

canvas.save(OUT)
print(
    f">>> 저장: {OUT}  ({canvas.width}x{canvas.height}, 타일 {len(tiles)})", flush=True
)
