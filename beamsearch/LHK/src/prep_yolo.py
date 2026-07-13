import json
import os
from pathlib import Path
from collections import defaultdict
import numpy as np

BASE = Path("/Users/macbook/dev/learning/codeit/01_Proj_HealthEat_cla")
DATA = BASE / "01_data/01_sprint_ai_project1_data"
TA, TI = DATA / "train_annotations", DATA / "train_images"
LHK = BASE / "project1-3team/beamsearch/LHK"
SSOT = LHK / "data/processed"
OUT = LHK / "data/yolo"
SEED = 42

cm = json.load(open(SSOT / "class_map.json", encoding="utf-8"))
c2m = {int(k): v for k, v in cm["category_id_to_model_index"].items()}
m2c = {int(k): v for k, v in cm["model_index_to_category_id"].items()}
NC = cm["num_classes"]

# 1) pill-JSON → 이미지 단위 재조립
img_anns = defaultdict(list)
img_wh = {}
for jf in TA.rglob("*.json"):
    d = json.load(open(jf, encoding="utf-8"))
    im = d["images"][0]
    fn = im["file_name"]
    img_wh[fn] = (im["width"], im["height"])
    for a in d["annotations"]:
        img_anns[fn].append((int(a["category_id"]), a["bbox"]))
files = sorted(img_anns)

# 2) 조합 단위 GroupKFold(5) → fold0 = val
sets = {fn: frozenset(c for c, _ in img_anns[fn]) for fn in files}
combo_of = {c: i for i, c in enumerate(dict.fromkeys(sets[fn] for fn in files))}
groups = np.array([combo_of[sets[fn]] for fn in files])
rng = np.random.default_rng(SEED)
uniq = np.array(sorted(set(groups)))
rng.shuffle(uniq)
folds = np.array_split(uniq, 5)
g2f = {int(g): k for k, fl in enumerate(folds) for g in fl}
split = {
    fn: ("val" if g2f[int(groups[i])] == 0 else "train") for i, fn in enumerate(files)
}

# 3) 라벨 + 이미지 심링크
for sp in ("train", "val"):
    (OUT / "images" / sp).mkdir(parents=True, exist_ok=True)
    (OUT / "labels" / sp).mkdir(parents=True, exist_ok=True)
cnt = defaultdict(int)
nboxes = defaultdict(int)
bad = 0
val_cls = set()
for i, fn in enumerate(files):
    sp = split[fn]
    W, H = img_wh[fn]
    dst = OUT / "images" / sp / fn
    if not dst.exists():
        os.symlink(TI / fn, dst)
    lines = []
    for cid, (x, y, w, h) in img_anns[fn]:
        m = c2m[cid]
        cx, cy, nw, nh = (x + w / 2) / W, (y + h / 2) / H, w / W, h / H
        if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < nw <= 1 and 0 < nh <= 1):
            bad += 1
        lines.append(f"{m} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
        if sp == "val":
            val_cls.add(m)
    (OUT / "labels" / sp / (Path(fn).stem + ".txt")).write_text("\n".join(lines))
    cnt[sp] += 1
    nboxes[sp] += len(lines)

# 4) data.yaml (names = model_index -> category_id 문자열, ASCII 안전)
names = "\n".join(f"  {i}: '{m2c[i]}'" for i in range(NC))
yaml = f"""# LHK E1 baseline (train-56, GroupKFold fold0=val) — auto-generated
path: {OUT}
train: images/train
val: images/val
nc: {NC}
names:
{names}
"""
(OUT / "data.yaml").write_text(yaml, encoding="utf-8")

print(f"train img {cnt['train']} / boxes {nboxes['train']}")
print(
    f"val   img {cnt['val']} / boxes {nboxes['val']}  | val 클래스 {len(val_cls)}/{NC}"
)
print(f"정규화 벗어난 box: {bad}")
print(f"data.yaml → {OUT / 'data.yaml'}")
# 샘플 라벨
sf = next((OUT / "labels" / "train").glob("*.txt"))
print(f"\n샘플 라벨 {sf.name}:\n{sf.read_text()}")
