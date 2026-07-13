"""RT-DETR Colab용 데이터 번들: E3와 동일 fold0(real181+synth3196) YOLO셋 + fold0 val 하니스 자산 → zip."""

import json
import os
import zipfile
from pathlib import Path

BASE = Path("/Users/macbook/dev/learning/codeit/01_Proj_HealthEat_cla")
LHK = BASE / "project1-3team/beamsearch/LHK"
AUG = (
    LHK / "data/yolo_aug_combined"
)  # E3가 만든 real181+synth3196 train + 51 val (심링크)
SSOT = LHK / "data/processed"
OUTZIP = BASE / "rtdetr_data_bundle.zip"
W, H = 976, 1280
assert AUG.exists(), "yolo_aug_combined 없음 (E3 실행이 먼저 구성함)"

cm = json.load(open(SSOT / "class_map.json"))
m2c = {int(k): v for k, v in cm["model_index_to_category_id"].items()}

# fold0 val GT (dl_idx, image_id=val 정렬순 1..51) — YOLO E3와 동일 하니스
val_imgs = sorted(
    p for p in (AUG / "images/val").iterdir() if not p.name.startswith(".")
)
nid = {p.name: i + 1 for i, p in enumerate(val_imgs)}
gt = {
    "images": [],
    "annotations": [],
    "categories": [{"id": c} for c in sorted(set(m2c.values()))],
}
aid = 1
for p in val_imgs:
    iid = nid[p.name]
    gt["images"].append({"id": iid, "file_name": p.name, "width": W, "height": H})
    for ln in (AUG / "labels/val" / (p.stem + ".txt")).read_text().splitlines():
        if not ln.strip():
            continue
        m, cx, cy, nw, nh = ln.split()
        m = int(m)
        cx, cy, nw, nh = map(float, (cx, cy, nw, nh))
        gt["annotations"].append(
            {
                "id": aid,
                "image_id": iid,
                "category_id": m2c[m],
                "bbox": [(cx - nw / 2) * W, (cy - nh / 2) * H, nw * W, nh * H],
                "area": nw * W * nh * H,
                "iscrowd": 0,
            }
        )
        aid += 1

names = "\n".join(f"  {i}: '{m2c[i]}'" for i in range(56))
data_yaml = f"path: .\ntrain: yolo/images/train\nval: yolo/images/val\nnc: 56\nnames:\n{names}\n"

if OUTZIP.exists():
    OUTZIP.unlink()
n_img = 0
with zipfile.ZipFile(
    OUTZIP, "w", zipfile.ZIP_STORED
) as z:  # jpg/png 이미 압축 → STORED(빠름)
    for sp in ("train", "val"):
        for p in (AUG / "images" / sp).iterdir():
            if p.name.startswith("."):
                continue
            z.write(
                os.path.realpath(p), f"yolo/images/{sp}/{p.name}"
            )  # 심링크 타깃 실내용
            n_img += 1
        for p in (AUG / "labels" / sp).iterdir():
            if p.name.endswith(".txt"):
                z.write(p, f"yolo/labels/{sp}/{p.name}")
    z.writestr("yolo/data.yaml", data_yaml)
    z.writestr("harness/val_gt.json", json.dumps(gt))
    z.writestr("harness/class_map.json", json.dumps(cm))
print(
    f"번들: {OUTZIP}  ({OUTZIP.stat().st_size / 1e6:.0f} MB, 이미지 {n_img}장, val {len(val_imgs)})"
)
