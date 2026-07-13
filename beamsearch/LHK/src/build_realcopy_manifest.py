"""real Copy-Paste 핸드오프용 소스 매니페스트: fold0 TRAIN(181장, 누수안전)의 알약 bbox+class만 추출.
Codex는 이 매니페스트에 나열된 이미지·박스만 소스로 사용 → val/test 알약 유입(누수) 원천 차단."""

import json
from pathlib import Path
from collections import Counter

LHK = Path(
    "/Users/macbook/dev/learning/codeit/01_Proj_HealthEat_cla/project1-3team/beamsearch/LHK"
)
YD, SSOT = LHK / "data/yolo", LHK / "data/processed"
OUT = LHK / "handoff_realcopy"
OUT.mkdir(exist_ok=True)
W, H = 976, 1280

cm = json.load(open(SSOT / "class_map.json"))
m2c = {int(k): v for k, v in cm["model_index_to_category_id"].items()}

train_imgs = sorted((YD / "images/train").glob("*.png"))
images = []
cls_count = Counter()
ppi = []
for p in train_imgs:
    pills = []
    for ln in (YD / "labels/train" / (p.stem + ".txt")).read_text().splitlines():
        if not ln.strip():
            continue
        mi, cx, cy, nw, nh = ln.split()
        mi = int(mi)
        cx, cy, nw, nh = map(float, (cx, cy, nw, nh))
        cid = m2c[mi]
        cls_count[cid] += 1
        pills.append(
            {
                "category_id": cid,
                "bbox_px": [
                    round((cx - nw / 2) * W, 1),
                    round((cy - nh / 2) * H, 1),
                    round(nw * W, 1),
                    round(nh * H, 1),
                ],
            }
        )
    ppi.append(len(pills))
    images.append({"file": p.name, "pills": pills})

manifest = {
    "_about": "fold0 TRAIN 알약 소스 (real Copy-Paste 전용). 여기 나열된 이미지·박스만 사용할 것.",
    "image_dir_macstudio": "/Volumes/SSD 4T/01_sprint_ai_project1_data/train_images",
    "image_size": [W, H],
    "coord_format": "bbox_px = [x, y, w, h] in pixels (좌상단 기준)",
    "leakage_rule": f"fold0 train {len(train_imgs)}장만 포함. val 51장·test는 의도적으로 제외(누수 차단).",
    "source_pill_instances": sum(ppi),
    "class_count_natural": dict(sorted(cls_count.items())),
    "pills_per_image": {
        "min": min(ppi),
        "max": max(ppi),
        "mean": round(sum(ppi) / len(ppi), 2),
    },
    "images": images,
}
(OUT / "realcopy_source_manifest_fold0train.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=1)
)

# 참고용: 696 categories 스키마(출력 포맷 패리티) 미러
synth696 = Path(
    "/Users/macbook/dev/learning/codeit/01_Proj_HealthEat_cla/01_data/processed/kaggle_sam2_synth_v2_kaggle_696/coco/annotations_coco.json"
)
cats = json.load(open(synth696))["categories"] if synth696.exists() else []
(OUT / "target_categories_schema.json").write_text(
    json.dumps(cats, ensure_ascii=False, indent=1)
)

print(f"매니페스트: {OUT / 'realcopy_source_manifest_fold0train.json'}")
print(
    f"  이미지 {len(images)} | 소스 알약 {sum(ppi)} | 클래스 {len(cls_count)} | 알약/img {min(ppi)}~{max(ppi)}(mean {sum(ppi) / len(ppi):.2f})"
)
print(
    f"  희소(≤3): {sum(1 for v in cls_count.values() if v <= 3)}클래스 | 최다 {max(cls_count.values())} 최소 {min(cls_count.values())}"
)
print(
    f"categories 스키마 미러: {OUT / 'target_categories_schema.json'} ({len(cats)} cats)"
)
if cats:
    print("  cat 예시:", json.dumps(cats[0], ensure_ascii=False))
