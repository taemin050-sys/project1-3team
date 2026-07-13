"""RT-DETR Colab 최종 번들: real232(corrected+8) + synth696 + aihub_full(inpaint) → YOLO train
+ test842 + class_map + data.yaml → zip. 학습이미지는 long side 960 다운스케일(정규화 라벨이라 무손실),
test는 원본 유지. Colab에서 RT-DETR-l 학습 → test 예측 → 제출 CSV."""

import json
import os
import sys
import shutil
import zipfile
from pathlib import Path
from collections import defaultdict

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths

TA, TI, TEST, PROCESSED = (
    paths.TRAIN_ANNOTATIONS,
    paths.TRAIN_IMAGES,
    paths.TEST_IMAGES,
    paths.PROCESSED,
)
SSOT = paths.SSOT
STAGE = paths.LHK / "data/rtdetr_bundle"
ZIP = paths.LHK / "data" / os.environ.get("EXP_BUNDLE", "rtdetr_final_bundle.zip")
SYNTH = [
    s
    for s in os.environ.get(
        "EXP_SYNTH", "kaggle_sam2_synth_v2_kaggle_696,kaggle_aihub_full_inpaint"
    ).split(",")
    if s
]  # 커버리지: EXP_SYNTH=kaggle_sam2_synth_v2_kaggle_696,kaggle_aihub_cover116
LONG = 960  # 학습이미지 long side 다운스케일

# 라벨정리: EXP_EXCLUDE=제외목록(basename 한 줄씩) → real+synth에서 제외(=clean246 반영).
EXCLUDE = set()
_exf = os.environ.get("EXP_EXCLUDE", "").strip()
if _exf:
    _exp = Path(_exf)
    if not _exp.is_absolute():
        _exp = Path(__file__).resolve().parent / _exf
    if not _exp.exists():
        sys.exit(f"[rtdetr_bundle] EXP_EXCLUDE 파일 없음: {_exp}")
    EXCLUDE = {ln.strip() for ln in _exp.read_text().splitlines() if ln.strip()}
    print(f"라벨정리: 제외목록 {_exp.name} → {len(EXCLUDE)}장 제외", flush=True)
cm = json.load(open(SSOT / os.environ.get("EXP_CLASSMAP", "class_map.json")))
c2m = {int(k): v for k, v in cm["category_id_to_model_index"].items()}
m2c = {int(k): v for k, v in cm["model_index_to_category_id"].items()}
NC = cm["num_classes"]

if STAGE.exists():
    shutil.rmtree(STAGE)
(STAGE / "images/train").mkdir(parents=True)
(STAGE / "labels/train").mkdir(parents=True)
(STAGE / "test").mkdir(parents=True)


def save_ds(name, W, H, boxes_norm, img_bgr):
    """정규화 박스 + 이미지 → long side 960 다운스케일 저장."""
    h, w = img_bgr.shape[:2]
    s = LONG / max(h, w)
    if s < 1:
        img_bgr = cv2.resize(
            img_bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA
        )
    cv2.imwrite(
        str(STAGE / "images/train" / (name + ".jpg")),
        img_bgr,
        [cv2.IMWRITE_JPEG_QUALITY, 90],
    )
    (STAGE / "labels/train" / (name + ".txt")).write_text(
        "\n".join(
            f"{m} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"
            for m, cx, cy, nw, nh in boxes_norm
        )
    )


# ---------- 1) real 232 (corrected +8) ----------
img_anns, img_wh = defaultdict(list), {}
for jf in TA.rglob("*.json"):
    d = json.load(open(jf))
    im = d["images"][0]
    img_wh[im["file_name"]] = (im["width"], im["height"])
    for a in d["annotations"]:
        img_anns[im["file_name"]].append((int(a["category_id"]), a["bbox"]))
corr = paths.LHK / "data/gt_corrected/corrections.json"
ncorr = 0
if corr.exists():
    for fn, adds in json.load(open(corr)).get("type_A_autofill", {}).items():
        for a in adds:
            img_anns[fn].append((int(a["category_id"]), a["bbox"]))
            ncorr += 1
n_real = n_exc = 0
for fn, anns in img_anns.items():
    if fn in EXCLUDE:  # 라벨정리: 의심 real 제외
        n_exc += 1
        continue
    W, H = img_wh[fn]
    img = cv2.imread(str(TI / fn))
    if img is None:
        continue
    bn = [
        (c2m[c], (x + bw / 2) / W, (y + bh / 2) / H, bw / W, bh / H)
        for c, (x, y, bw, bh) in anns
    ]
    save_ds(f"real_{Path(fn).stem}", W, H, bn, img)
    n_real += 1

# ---------- 2) synth + aihub ----------
n_syn = 0
for sd in SYNTH:
    coco = json.load(open(PROCESSED / sd / "coco/annotations_coco.json"))
    anns_by = defaultdict(list)
    for a in coco["annotations"]:
        anns_by[a["image_id"]].append(a)
    for im in coco["images"]:
        if im["file_name"] in EXCLUDE:  # 라벨정리: 의심 synth/aihub 제외
            n_exc += 1
            continue
        src = PROCESSED / sd / "coco/images" / im["file_name"]
        if not src.exists():
            continue
        img = cv2.imread(str(src))
        if img is None:
            continue
        iw, ih = im["width"], im["height"]
        bn = [
            (
                c2m[int(a["category_id"])],
                (a["bbox"][0] + a["bbox"][2] / 2) / iw,
                (a["bbox"][1] + a["bbox"][3] / 2) / ih,
                a["bbox"][2] / iw,
                a["bbox"][3] / ih,
            )
            for a in anns_by[im["id"]]
        ]
        save_ds(f"{sd[:8]}_{Path(im['file_name']).stem}", iw, ih, bn, img)
        n_syn += 1

# ---------- 3) test 842 (원본) + configs ----------
n_test = 0
for p in TEST.glob("*.png"):
    shutil.copy(p, STAGE / "test" / p.name)
    n_test += 1
names = "\n".join(f"  {i}: '{m2c[i]}'" for i in range(NC))
(STAGE / "data.yaml").write_text(
    f"path: .\ntrain: images/train\nval: images/train\nnc: {NC}\nnames:\n{names}\n"
)
json.dump(cm, open(STAGE / "class_map.json", "w"))
print(
    f"스테이징: real {n_real}(corrected +{ncorr}) + synth/aihub {n_syn} = train {n_real + n_syn} (제외 {n_exc}) | test {n_test}",
    flush=True,
)

# ---------- 4) zip ----------
if ZIP.exists():
    ZIP.unlink()
with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_STORED) as z:
    for f in STAGE.rglob("*"):
        if f.is_file():
            z.write(f, f.relative_to(STAGE))
print(f">>> 번들: {ZIP} ({ZIP.stat().st_size / 1e9:.2f} GB)", flush=True)
