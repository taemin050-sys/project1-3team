"""per-fold real-copy 소스 매니페스트 (누수 차단). 각 폴드 K에 대해 fold-K-train 조합의 알약만 소스.
k-fold 평가 시 fold K의 real-copy는 fold-K-train에서만 뽑아 → 그 폴드 val 알약이 train에 안 샘(누수0).
출력: handoff_realcopy/realcopy_src_fold{K}train.json (Codex가 폴드별 real-copy 생성용)."""

import json
import os
import sys
from collections import defaultdict, Counter
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths

TA = paths.TRAIN_ANNOTATIONS
OUT = paths.LHK / "handoff_realcopy"
OUT.mkdir(exist_ok=True)
cm = json.load(open(paths.SSOT / "class_map.json"))
name_of = {}
sc = paths.LHK / "handoff_realcopy/target_categories_schema.json"
if sc.exists():
    name_of = {c["id"]: c.get("name", "") for c in json.load(open(sc))}

# 이미지별 GT + 폴드 (prep_yolo 동일)
img_boxes, img_wh, img_cats = defaultdict(list), {}, defaultdict(list)
for jf in TA.rglob("*.json"):
    d = json.load(open(jf))
    im = d["images"][0]
    img_wh[im["file_name"]] = (im["width"], im["height"])
    for a in d["annotations"]:
        img_boxes[im["file_name"]].append(
            (int(a["category_id"]), [round(float(v), 1) for v in a["bbox"]])
        )
        img_cats[im["file_name"]].append(int(a["category_id"]))
files = sorted(img_boxes)
sets = {fn: frozenset(img_cats[fn]) for fn in files}
combo_of = {c: i for i, c in enumerate(dict.fromkeys(sets[fn] for fn in files))}
groups = np.array([combo_of[sets[fn]] for fn in files])
rng = np.random.default_rng(42)
uniq = np.array(sorted(set(groups)))
rng.shuffle(uniq)
fold_groups = np.array_split(uniq, 5)
g2f = {int(g): k for k, fl in enumerate(fold_groups) for g in fl}
img_fold = {fn: g2f[int(groups[i])] for i, fn in enumerate(files)}

for K in (0, 1, 2):
    train = [fn for fn in files if img_fold[fn] != K]
    images, cc, ppi = [], Counter(), []
    for fn in train:
        pills = [{"category_id": c, "bbox_px": b} for c, b in img_boxes[fn]]
        for c, _ in img_boxes[fn]:
            cc[c] += 1
        ppi.append(len(pills))
        images.append({"file": fn, "pills": pills})
    man = {
        "_about": f"fold{K} real-copy 소스 — fold{K}-train 조합만 (fold{K} val 조합 제외 → 누수0).",
        "fold": K,
        "image_dir_macstudio": "/Volumes/SSD 4T/01_sprint_ai_project1_data/train_images",
        "image_size": [976, 1280],
        "coord_format": "bbox_px = [x,y,w,h] px",
        "leakage_rule": f"fold{K}-train {len(train)}장만. fold{K} val 조합 의도적 제외.",
        "source_pill_instances": sum(ppi),
        "class_count_natural": {str(k): v for k, v in sorted(cc.items())},
        "pills_per_image": {
            "min": min(ppi),
            "max": max(ppi),
            "mean": round(sum(ppi) / len(ppi), 2),
        },
        "images": images,
    }
    fp = OUT / f"realcopy_src_fold{K}train.json"
    fp.write_text(json.dumps(man, ensure_ascii=False))
    print(
        f"fold{K}: train {len(train)}장 / 소스알약 {sum(ppi)} / 클래스 {len(cc)} → {fp.name}"
    )
print(f"\n출력 위치: {OUT}")
