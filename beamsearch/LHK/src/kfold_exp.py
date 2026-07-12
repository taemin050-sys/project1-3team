"""파라미터화된 k-fold 실험 러너 — 단일변수 실험을 환경변수로 재사용.
  EXP_VARIANT  결과에 남길 변주명 (예: s696_11s, s696_bg64)
  EXP_MODEL    ultralytics 가중치 (기본 yolo11n.pt)
  EXP_IMGSZ    학습·추론 해상도 (기본 640)
  EXP_SYNTH    train에 추가할 synth 디렉토리(PROCESSED 하위), 콤마구분. 'none'=real만
  EXP_FOLDS    폴드 (기본 0,1,2)
  EXP_EXCLUDE  라벨정리 실험용 제외목록 파일(이미지 basename 한 줄씩). train real+synth에서만 제외, val 고정.
예) EXP_VARIANT=s696_11s EXP_MODEL=yolo11s.pt EXP_SYNTH=kaggle_sam2_synth_v2_kaggle_696 \
     bash scripts/run.sh scripts/kfold_exp.py cap11s
하이퍼는 baseline과 동일(50ep·b16·mps·seed42·patience20). 결과 → labkit 버스(머신별). skip-done 지원."""

import json
import os
import sys
import shutil
from pathlib import Path
from collections import defaultdict
import numpy as np
from ultralytics import YOLO
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths
import labkit

VARIANT = os.environ.get("EXP_VARIANT") or sys.exit("EXP_VARIANT 필수")
MODEL = os.environ.get("EXP_MODEL", "yolo11n.pt")
IMGSZ = int(os.environ.get("EXP_IMGSZ", "640"))
SYNTH = [s for s in os.environ.get("EXP_SYNTH", "").split(",") if s and s != "none"]
FOLDS = [int(x) for x in os.environ.get("EXP_FOLDS", "0,1,2").split(",")]
EPOCHS = int(
    os.environ.get("EXP_EPOCHS", "50")
)  # 큰 모델은 100 권장(patience로 수렴 시 조기종료)
SEED, W, H = 42, 976, 1280

# 증강 프로파일: EXP_AUG=strong → 강한 albumentations+기하 증강(저데이터 대응). default=ultralytics 기본.
AUG = os.environ.get("EXP_AUG", "default")
AUG_KW = {}
if AUG == "strong":
    AUG_KW = dict(
        degrees=15.0,
        translate=0.1,
        scale=0.5,
        shear=3.0,
        perspective=0.0005,
        flipud=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        erasing=0.5,
        auto_augment="randaugment",
    )  # ultralytics는 albumentations 설치 시 Blur/CLAHE/ToGray 등 자동 추가
elif AUG == "realistic":
    # 기본 + 전각 in-plane 회전만(평평한 알약은 어느 각도든 현실적). 도메인 깨는 flipud/perspective/shear/mixup 배제.
    AUG_KW = dict(degrees=180.0)

# 희소클래스 real 오버샘플링: EXP_OVERSAMPLE=floor(클래스당 목표 인스턴스). 0=off.
OVERSAMPLE = int(os.environ.get("EXP_OVERSAMPLE", "0"))

# 라벨정리 실험: EXP_EXCLUDE=제외목록파일(이미지 basename 한 줄씩). train에서만 제외(val 고정 → A/B 공정).
EXCLUDE = set()
_exf = os.environ.get("EXP_EXCLUDE", "").strip()
if _exf:
    _exp = Path(_exf)
    if not _exp.is_absolute():
        _exp = Path(__file__).resolve().parent / _exf
    if not _exp.exists():
        sys.exit(f"[kfold_exp] EXP_EXCLUDE 파일 없음: {_exp}")
    EXCLUDE = {ln.strip() for ln in _exp.read_text().splitlines() if ln.strip()}
    print(
        f"라벨정리: 제외목록 {_exp.name} → {len(EXCLUDE)}장 (train에서만 제외)",
        flush=True,
    )

TA, TI, PROCESSED = paths.TRAIN_ANNOTATIONS, paths.TRAIN_IMAGES, paths.PROCESSED
SSOT, RUNS = paths.SSOT, paths.RUNS
WORK = paths.LHK / "data/kfold_work_exp"
KRUNS = RUNS / "kfold"
KRUNS.mkdir(parents=True, exist_ok=True)

# 프리플라이트: synth 디렉토리 존재 확인 ({fold}은 각 폴드로 치환해 전부 확인)
for sd in [
    s.format(fold=f) for s in SYNTH for f in (FOLDS if "{fold}" in s else FOLDS[:1])
]:
    if not (PROCESSED / sd / "coco/annotations_coco.json").exists():
        avail = (
            sorted(p.name for p in PROCESSED.iterdir() if p.is_dir())
            if PROCESSED.exists()
            else []
        )
        sys.exit(
            f"[kfold_exp] synth '{sd}' 없음 (PROCESSED={PROCESSED}).\n  가용: {avail}\n  → EXP_SYNTH 를 실제 디렉토리명으로."
        )

print(
    f"실험: variant={VARIANT} model={MODEL} imgsz={IMGSZ} epochs={EPOCHS} synth={SYNTH or 'real만'} folds={FOLDS} machine={paths.MACHINE}",
    flush=True,
)

cm = json.load(
    open(SSOT / os.environ.get("EXP_CLASSMAP", "class_map.json"), encoding="utf-8")
)
c2m = {int(k): v for k, v in cm["category_id_to_model_index"].items()}
m2c = {int(k): v for k, v in cm["model_index_to_category_id"].items()}
NC = cm["num_classes"]

img_anns, img_wh = defaultdict(list), {}
for jf in TA.rglob("*.json"):
    d = json.load(open(jf, encoding="utf-8"))
    im = d["images"][0]
    img_wh[im["file_name"]] = (im["width"], im["height"])
    for a in d["annotations"]:
        img_anns[im["file_name"]].append((int(a["category_id"]), a["bbox"]))
files = sorted(img_anns)
sets = {fn: frozenset(c for c, _ in img_anns[fn]) for fn in files}
combo_of = {c: i for i, c in enumerate(dict.fromkeys(sets[fn] for fn in files))}
groups = np.array([combo_of[sets[fn]] for fn in files])
rng = np.random.default_rng(SEED)
uniq = np.array(sorted(set(groups)))
rng.shuffle(uniq)
fold_groups = np.array_split(uniq, 5)
g2f = {int(g): k for k, fl in enumerate(fold_groups) for g in fl}
img_fold = {fn: g2f[int(groups[i])] for i, fn in enumerate(files)}


def build_split(fold):
    if WORK.exists():
        shutil.rmtree(WORK)
    for sp in ("train", "val"):
        (WORK / "images" / sp).mkdir(parents=True)
        (WORK / "labels" / sp).mkdir(parents=True)
    ntr = nval = nexc = 0
    for fn in files:
        sp = "val" if img_fold[fn] == fold else "train"
        if sp == "train" and fn in EXCLUDE:  # 라벨정리: train real 제외(val 고정)
            nexc += 1
            continue
        w, h = img_wh[fn]
        os.symlink(TI / fn, WORK / "images" / sp / fn)
        (WORK / "labels" / sp / (Path(fn).stem + ".txt")).write_text(
            "\n".join(
                f"{c2m[c]} {(x + bw / 2) / w:.6f} {(y + bh / 2) / h:.6f} {bw / w:.6f} {bh / h:.6f}"
                for c, (x, y, bw, bh) in img_anns[fn]
            )
        )
        nval += sp == "val"
        ntr += sp == "train"
    # 희소클래스 real 오버샘플링 (그 폴드 train 이미지만 복제 → 누수0, 100% real)
    n_over = 0
    if OVERSAMPLE:
        real_tr = [fn for fn in files if img_fold[fn] != fold and fn not in EXCLUDE]
        cc, img_cls, by_cls = defaultdict(int), {}, defaultdict(list)
        for fn in real_tr:
            cls = [c2m[c] for c, _ in img_anns[fn]]
            img_cls[fn] = cls
            for m in cls:
                cc[m] += 1
            for m in set(cls):
                by_cls[m].append(fn)
        while n_over < 3000:
            deficient = [m for m in cc if cc[m] < OVERSAMPLE and by_cls[m]]
            if not deficient:
                break
            m = min(deficient, key=lambda x: cc[x])
            fn = by_cls[m][cc[m] % len(by_cls[m])]
            nm = f"{Path(fn).stem}_ov{n_over}"
            os.symlink(TI / fn, WORK / "images/train" / (nm + ".png"))
            shutil.copy(
                WORK / "labels/train" / (Path(fn).stem + ".txt"),
                WORK / "labels/train" / (nm + ".txt"),
            )
            for mm in img_cls[fn]:
                cc[mm] += 1
            n_over += 1
        print(
            f"오버샘플: +{n_over}장 (floor={OVERSAMPLE}, 최소클래스 {min(cc.values())}/{NC}클래스)",
            flush=True,
        )
        ntr += n_over
    nsyn = 0
    for j, sd_raw in enumerate(SYNTH):
        sd = sd_raw.format(
            fold=fold
        )  # 누수안전 per-fold real-copy: 이름에 {fold} 넣으면 폴드별 디렉토리 사용
        coco = json.load(open(PROCESSED / sd / "coco/annotations_coco.json"))
        anns_by = defaultdict(list)
        for a in coco["annotations"]:
            anns_by[a["image_id"]].append(a)
        for im in coco["images"]:
            if im["file_name"] in EXCLUDE:  # 라벨정리: synth/aihub 의심 제외
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
    return ntr, nsyn, nval, nexc


def evaluate(best, fold):
    val = sorted(fn for fn in files if img_fold[fn] == fold)
    nid = {fn: i + 1 for i, fn in enumerate(val)}
    gt = {
        "images": [],
        "annotations": [],
        "categories": [{"id": c} for c in sorted(set(m2c.values()))],
    }
    aid = 1
    for fn in val:
        gt["images"].append({"id": nid[fn], "file_name": fn, "width": W, "height": H})
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
    gp = KRUNS / f"gt_exp_f{fold}.json"
    json.dump(gt, open(gp, "w"))
    cocoGt = COCO(str(gp))
    dts = []
    for fn, res in zip(
        val,
        best.predict(
            [str(TI / fn) for fn in val],
            conf=0.001,
            iou=0.6,
            max_det=100,
            imgsz=IMGSZ,
            device="mps",
            verbose=False,
        ),
    ):
        for b, c, s in zip(
            res.boxes.xyxy.cpu().numpy(),
            res.boxes.cls.cpu().numpy(),
            res.boxes.conf.cpu().numpy(),
        ):
            x1, y1, x2, y2 = b
            dts.append(
                {
                    "image_id": nid[fn],
                    "category_id": m2c[int(c)],
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "score": float(s),
                }
            )
    e = COCOeval(cocoGt, cocoGt.loadRes(dts), "bbox")
    e.params.iouThrs = np.linspace(0.75, 0.95, 5)
    e.evaluate()
    e.accumulate()
    e.summarize()
    return float(e.stats[0])


done = labkit.done_keys("healtheat", ["variant", "fold"])
for fold in FOLDS:
    if (VARIANT, fold) in done:
        print(f"[skip] {VARIANT} fold{fold}", flush=True)
        continue
    ntr, nsyn, nval, nexc = build_split(fold)
    print(
        f"\n=== {VARIANT} fold{fold}: real{ntr}+synth{nsyn} val{nval} exclude{nexc} | {MODEL} @imgsz{IMGSZ} ===",
        flush=True,
    )
    YOLO(MODEL).train(
        data=str(WORK / "data.yaml"),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=16,
        device="mps",
        seed=SEED,
        deterministic=True,
        workers=4,
        patience=20,
        project=str(KRUNS),
        name=f"f{fold}_{VARIANT}",
        exist_ok=True,
        plots=False,
        verbose=False,
        **AUG_KW,
    )
    best = YOLO(str(KRUNS / f"f{fold}_{VARIANT}/weights/best.pt"))
    mAP = evaluate(best, fold)
    labkit.record(
        "healtheat",
        {
            "variant": VARIANT,
            "fold": fold,
            "mAP_75_95": round(mAP, 4),
            "model": MODEL,
            "imgsz": IMGSZ,
            "epochs": EPOCHS,
            "n_train_real": ntr,
            "n_synth": nsyn,
            "n_val": nval,
            "n_excluded": nexc,
        },
    )
    print(f">>> {VARIANT} fold{fold}: mAP@[0.75:0.95] = {mAP:.4f}", flush=True)

xs = [
    r["mAP_75_95"] for r in labkit.load_records("healtheat") if r["variant"] == VARIANT
]
if xs:
    print(
        f"\n>>> {VARIANT} 폴드평균 = {np.mean(xs):.4f}  {[round(x, 4) for x in xs]}  (기준 s696@11n=0.905)",
        flush=True,
    )
