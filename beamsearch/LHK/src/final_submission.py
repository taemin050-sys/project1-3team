"""최종 제출용: real 232 전량 + synth696 으로 YOLO11s 학습(홀드아웃 없음) → test842 예측 → 제출 CSV.
로컬 k-fold 검증한 config(11s@640·synth696)를 전량 데이터로 학습 → 리더보드 캘리브레이션용.
val은 파이프라인용 소규모 서브셋(전량 학습이 목적이라 metric 무의미). predict=canonical(conf0.001/iou0.6)."""

import csv
import json
import os
import sys
import shutil
from pathlib import Path
from collections import defaultdict
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths

TA, TI, TEST, PROCESSED = (
    paths.TRAIN_ANNOTATIONS,
    paths.TRAIN_IMAGES,
    paths.TEST_IMAGES,
    paths.PROCESSED,
)
SSOT, RUNS = paths.SSOT, paths.RUNS
WORK = paths.LHK / "data/final_work"
OUT_DIR = RUNS / "final"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SYNTH = [
    s
    for s in os.environ.get(
        "EXP_SYNTH", "kaggle_sam2_synth_v2_kaggle_696,kaggle_aihub_full_inpaint"
    ).split(",")
    if s
]  # 커버리지 학습: EXP_SYNTH=kaggle_sam2_synth_v2_kaggle_696,kaggle_aihub_cover116
MODEL = os.environ.get("EXP_MODEL", "yolo11s.pt")  # 11m 등 모델 선택
IMGSZ, SEED = 640, 42
EPOCHS = int(os.environ.get("EXP_EPOCHS", "50"))  # 큰 모델은 100 권장(수렴 여유)

# 라벨정리: EXP_EXCLUDE=제외목록(basename 한 줄씩) → real+synth train에서 제외. EXP_TAG=출력 접미사(예 _clean246).
EXCLUDE = set()
_exf = os.environ.get("EXP_EXCLUDE", "").strip()
if _exf:
    _exp = Path(_exf)
    if not _exp.is_absolute():
        _exp = Path(__file__).resolve().parent / _exf
    if not _exp.exists():
        sys.exit(f"[final] EXP_EXCLUDE 파일 없음: {_exp}")
    EXCLUDE = {ln.strip() for ln in _exp.read_text().splitlines() if ln.strip()}
    print(
        f"라벨정리: 제외목록 {_exp.name} → {len(EXCLUDE)}장 (train에서 제외)",
        flush=True,
    )
TAG = os.environ.get("EXP_TAG", "")

cm = json.load(
    open(SSOT / os.environ.get("EXP_CLASSMAP", "class_map.json"), encoding="utf-8")
)
c2m = {int(k): v for k, v in cm["category_id_to_model_index"].items()}
m2c = {int(k): v for k, v in cm["model_index_to_category_id"].items()}
NC = cm["num_classes"]

# ---------- 1) 전량 train + synth696, val=소규모 서브셋(파이프라인용) ----------
img_anns, img_wh = defaultdict(list), {}
for jf in TA.rglob("*.json"):
    d = json.load(open(jf, encoding="utf-8"))
    im = d["images"][0]
    img_wh[im["file_name"]] = (im["width"], im["height"])
    for a in d["annotations"]:
        img_anns[im["file_name"]].append((int(a["category_id"]), a["bbox"]))
# corrected 라벨: 확정 누락 8건(Type-A) 보강 → 0.99 구간 FP 제거
_corr = paths.LHK / "data/gt_corrected/corrections.json"
if _corr.exists():
    _cj = json.load(open(_corr))
    _n = 0
    for _fn, _adds in _cj.get("type_A_autofill", {}).items():
        for _a in _adds:
            img_anns[_fn].append((int(_a["category_id"]), _a["bbox"]))
            _n += 1
    print(f"corrected 라벨 보강: +{_n}건 (Type-A)", flush=True)
files = sorted(img_anns)

if WORK.exists():
    shutil.rmtree(WORK)
for sp in ("train", "val"):
    (WORK / "images" / sp).mkdir(parents=True)
    (WORK / "labels" / sp).mkdir(parents=True)


def write_label(dst_dir, stem, fn):
    w, h = img_wh[fn]
    (dst_dir / (stem + ".txt")).write_text(
        "\n".join(
            f"{c2m[c]} {(x + bw / 2) / w:.6f} {(y + bh / 2) / h:.6f} {bw / w:.6f} {bh / h:.6f}"
            for c, (x, y, bw, bh) in img_anns[fn]
        )
    )


nexc = 0
for fn in files:  # 전량 232 → train
    if fn in EXCLUDE:  # 라벨정리: 의심 real 제외
        nexc += 1
        continue
    os.symlink(TI / fn, WORK / "images/train" / fn)
    write_label(WORK / "labels/train", Path(fn).stem, fn)
for fn in files[:40]:  # 소규모 val(파이프라인용, train과 겹침 — 최종모델이라 무관)
    os.symlink(TI / fn, WORK / "images/val" / fn)
    write_label(WORK / "labels/val", Path(fn).stem, fn)
nsyn = 0
for j, sd in enumerate(SYNTH):
    coco = json.load(open(PROCESSED / sd / "coco/annotations_coco.json"))
    anns_by = defaultdict(list)
    for a in coco["annotations"]:
        anns_by[a["image_id"]].append(a)
    for im in coco["images"]:
        if im["file_name"] in EXCLUDE:  # 라벨정리: 의심 synth/aihub 제외
            nexc += 1
            continue
        src = PROCESSED / sd / "coco/images" / im["file_name"]
        if not src.exists():
            continue
        name = f"s{j}_" + im["file_name"]
        os.symlink(os.path.realpath(src), WORK / "images/train" / name)
        iw, ih = im["width"], im["height"]
        (WORK / "labels/train" / (Path(name).stem + ".txt")).write_text(
            "\n".join(
                f"{c2m[int(a['category_id'])]} {(a['bbox'][0] + a['bbox'][2] / 2) / iw:.6f} {(a['bbox'][1] + a['bbox'][3] / 2) / ih:.6f} {a['bbox'][2] / iw:.6f} {a['bbox'][3] / ih:.6f}"
                for a in anns_by[im["id"]]
            )
        )
        nsyn += 1
names = "\n".join(f"  {i}: '{m2c[i]}'" for i in range(NC))
(WORK / "data.yaml").write_text(
    f"path: {WORK}\ntrain: images/train\nval: images/val\nnc: {NC}\nnames:\n{names}\n"
)
n_real = len(files) - sum(1 for f in files if f in EXCLUDE)
print(
    f"전량학습셋: real {n_real} + synth {nsyn} = {n_real + nsyn}장 (제외 {nexc}, 홀드아웃 없음)",
    flush=True,
)

# ---------- 2) 학습 ----------
RUN = f"{Path(MODEL).stem}_full232_synth696_aihubfull{TAG}"
YOLO(MODEL).train(
    data=str(WORK / "data.yaml"),
    epochs=EPOCHS,
    imgsz=IMGSZ,
    batch=16,
    device="mps",
    seed=SEED,
    deterministic=True,
    workers=4,
    patience=int(
        os.environ.get("EXP_PATIENCE", str(EPOCHS))
    ),  # 안전 캡(더미val이라 소프트)
    project=str(OUT_DIR),
    name=RUN,
    exist_ok=True,
    plots=False,
    verbose=False,
)
best = YOLO(str(OUT_DIR / RUN / "weights/best.pt"))

# ---------- 3) test842 예측 → 제출 CSV ----------
# per-image 예측: MPS는 전체 test 리스트를 한 배치로 처리하려다 대형 test셋에서
# 'MPSGraph does not support tensor dims larger than INT_MAX'로 죽음. 1장씩 처리해 회피,
# 실패 이미지는 CPU로 폴백. 결과·CSV 포맷은 배치 예측과 동일.
test_imgs = sorted(TEST.glob("*.png"), key=lambda p: int(p.stem))
rows, aid, ncpu = [], 1, 0
_pkw = dict(conf=0.001, iou=0.6, max_det=30, imgsz=IMGSZ, verbose=False)
for p in test_imgs:
    iid = int(p.stem)
    try:
        res = best.predict(str(p), device="mps", **_pkw)[0]
    except RuntimeError:
        res = best.predict(str(p), device="cpu", **_pkw)[0]
        ncpu += 1
    for b, c, s in zip(
        res.boxes.xyxy.cpu().numpy(),
        res.boxes.cls.cpu().numpy(),
        res.boxes.conf.cpu().numpy(),
    ):
        x1, y1, x2, y2 = b
        rows.append(
            [
                aid,
                iid,
                m2c[int(c)],
                round(float(x1), 1),
                round(float(y1), 1),
                round(float(x2 - x1), 1),
                round(float(y2 - y1), 1),
                round(float(s), 4),
            ]
        )
        aid += 1
if ncpu:
    print(f"[predict] MPS 실패 {ncpu}장 → CPU 폴백", flush=True)
sub = OUT_DIR / f"submission_{RUN}.csv"
with open(sub, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(
        [
            "annotation_id",
            "image_id",
            "category_id",
            "bbox_x",
            "bbox_y",
            "bbox_w",
            "bbox_h",
            "score",
        ]
    )
    w.writerows(rows)
print(f"\n>>> 제출 CSV: {sub}", flush=True)
print(
    f">>> rows={len(rows)} images={len({r[1] for r in rows})}/842 cats={len({r[2] for r in rows})}/{NC} "
    f"cat⊆{NC}={ {r[2] for r in rows} <= set(m2c.values()) }",
    flush=True,
)
