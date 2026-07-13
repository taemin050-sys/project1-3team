"""116-class 커버리지 추출: AI Hub 허가 조합의 '모든 알약'을 인페인트·필터 없이 전량 추출.
새 대회([AI12])는 클래스가 71~79로 늘어 56만 잡으면 점수 급락 → 가용 116클래스 전부 학습이 정답.
출력: PROCESSED/kaggle_aihub_cover116/coco/{annotations_coco.json, images/*.jpg}. category_id=약품코드(dl).
env: EXP_CAP=combo 상한(스모크용, 기본 전량). JPEG q95 저장."""

import io
import json
import os
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths

OUT = paths.PROCESSED / "kaggle_aihub_cover116"
IMGDIR = OUT / "coco/images"
IMGDIR.mkdir(parents=True, exist_ok=True)
ALLOW = Path("/Volumes/USB 1T/사용 허가 조합/01.데이터/1.Training")
TSD = ALLOW / "원천데이터/경구약제조합 5000종"
TLD = ALLOW / "라벨링데이터/경구약제조합 5000종"
TS_Z = ["TS_1", "TS_3", "TS_4", "TS_5", "TS_6", "TS_7", "TS_8"]  # 금지 TS_2 제외
TL_Z = ["TL_1", "TL_3", "TL_4", "TL_5", "TL_6", "TL_7", "TL_8"]
CAP = int(os.environ.get("EXP_CAP", "0"))  # 0=전량

# ---------- combo → (TS zip, [img members]) / TL zip ----------
tsz = {
    z: zipfile.ZipFile(TSD / f"{z}_조합.zip")
    for z in TS_Z
    if (TSD / f"{z}_조합.zip").exists()
}
tlz = {
    z: zipfile.ZipFile(TLD / f"{z}_조합.zip")
    for z in TL_Z
    if (TLD / f"{z}_조합.zip").exists()
}
combo_ts = {}  # combo -> (zipname, [members])
for z, zf in tsz.items():
    for n in zf.namelist():
        if n.endswith(".png") and "/" in n and "_index" not in n:
            combo_ts.setdefault(n.split("/")[0], (z, []))[1].append(n)
combo_tl = {}  # combo -> zipname
for z, zf in tlz.items():
    for n in zf.namelist():
        if "_json/" in n:
            combo_tl.setdefault(n.split("_json/")[0], z)

combos = sorted(combo_ts)
if CAP:
    combos = combos[:CAP]
print(
    f"허가 조합 {len(combos)}개 처리 시작 (인페인트·필터 없음, 전 클래스 보존)",
    flush=True,
)


def boxes_for(combo, base):
    z = combo_tl.get(combo)
    if not z:
        return []
    zf = tlz[z]
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


images, annots, cats = [], [], set()
iid = aid = 0
skip = 0
for ci, combo in enumerate(combos):
    z, members = combo_ts[combo]
    zf = tsz[z]
    for mem in members:
        base = Path(mem).stem
        bs = boxes_for(combo, base)
        if not bs:
            skip += 1
            continue
        img = cv2.imdecode(np.frombuffer(zf.read(mem), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            skip += 1
            continue
        H, W = img.shape[:2]
        fn = f"aihub_{base}.jpg"
        cv2.imwrite(str(IMGDIR / fn), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        iid += 1
        images.append({"id": iid, "file_name": fn, "width": W, "height": H})
        for d, (x, y, w, h) in bs:
            aid += 1
            cats.add(d)
            annots.append(
                {
                    "id": aid,
                    "image_id": iid,
                    "category_id": d,
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                }
            )
    if (ci + 1) % 200 == 0:
        print(
            f"  {ci + 1}/{len(combos)} combo · 이미지 {iid} · 박스 {aid} · 클래스 {len(cats)}",
            flush=True,
        )

coco = {
    "images": images,
    "annotations": annots,
    "categories": [{"id": c} for c in sorted(cats)],
}
json.dump(coco, open(OUT / "coco/annotations_coco.json", "w"))
print(f"\n>>> 완료: {OUT}", flush=True)
print(f"    이미지 {iid} · 박스 {aid} · 클래스 {len(cats)} · 스킵 {skip}", flush=True)
