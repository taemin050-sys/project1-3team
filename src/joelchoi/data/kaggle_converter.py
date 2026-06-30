"""Kaggle 대회 train_annotations → COCO 통합 포맷 변환."""

import json
from collections import defaultdict
from pathlib import Path


KAGGLE_DATA_ROOT = (
    Path.home()
    / ".cache/kagglehub/competitions/ai12-level1-project/sprint_ai_project1_data"
)


def find_kaggle_root(base_path: str | Path | None = None) -> Path:
    if base_path:
        return Path(base_path)
    if KAGGLE_DATA_ROOT.exists():
        return KAGGLE_DATA_ROOT
    raise FileNotFoundError(
        "Kaggle 데이터 경로를 찾을 수 없습니다. base_path를 지정해주세요."
    )


def convert_kaggle_to_coco(base_path: str | Path | None = None) -> dict:
    """Kaggle 대회 train 데이터 → 통합 COCO dict 변환.

    train_annotations/의 JSON 파일들을 파싱하여 하나의 COCO dict로 합칩니다.
    category_id는 드러그 코드의 숫자 부분 (K-001900 → 1900).

    Returns:
        COCO format dict (images, annotations, categories)
    """
    root = find_kaggle_root(base_path)
    ann_root = root / "train_annotations"
    img_root = root / "train_images"

    # {file_name: {width, height, path}}
    img_meta: dict[str, dict] = {}
    # {file_name: [ann_dict, ...]}
    img_anns: dict[str, list[dict]] = defaultdict(list)
    # {category_id: name}
    cat_info: dict[int, str] = {}

    for combo_dir in sorted(ann_root.iterdir()):
        if not combo_dir.is_dir() or not combo_dir.name.endswith("_json"):
            continue

        for drug_dir in sorted(combo_dir.iterdir()):
            if not drug_dir.is_dir():
                continue
            drug_code = drug_dir.name  # e.g., "K-001900"
            try:
                cat_id = int(drug_code.replace("K-", ""))
            except ValueError:
                continue

            for json_file in sorted(drug_dir.glob("*.json")):
                try:
                    data = json.loads(json_file.read_text(encoding="utf-8"))
                except Exception:
                    try:
                        data = json.loads(json_file.read_text(encoding="cp949", errors="replace"))
                    except Exception:
                        continue

                img_info = data["images"][0]
                file_name = img_info["file_name"]

                if file_name not in img_meta:
                    full_path = img_root / file_name
                    img_meta[file_name] = {
                        "width": img_info["width"],
                        "height": img_info["height"],
                        "path": str(full_path),
                    }

                # 카테고리 이름 수집
                for cat in data.get("categories", []):
                    if cat["id"] == cat_id and cat_id not in cat_info:
                        cat_info[cat_id] = cat["name"]

                for ann in data.get("annotations", []):
                    bbox = ann.get("bbox", [])
                    if not (isinstance(bbox, list) and len(bbox) == 4):
                        continue
                    if not all(isinstance(v, (int, float)) for v in bbox):
                        continue
                    img_anns[file_name].append({
                        "bbox": [float(v) for v in bbox],
                        "category_id": cat_id,
                        "area": float(ann.get("area", bbox[2] * bbox[3])),
                        "iscrowd": int(ann.get("iscrowd", 0)),
                    })

    # 카테고리 ID 순으로 정렬
    sorted_cats = sorted(cat_info.items(), key=lambda x: x[0])
    categories = [{"id": cid, "name": name} for cid, name in sorted_cats]

    images = []
    annotations = []
    img_id = 0
    ann_id = 0

    for file_name in sorted(img_meta.keys()):
        img_id += 1
        meta = img_meta[file_name]
        images.append({
            "id": img_id,
            "file_name": meta["path"],
            "width": meta["width"],
            "height": meta["height"],
        })

        for ann in img_anns.get(file_name, []):
            ann_id += 1
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": ann["category_id"],
                "bbox": ann["bbox"],
                "area": ann["area"],
                "iscrowd": ann["iscrowd"],
                "segmentation": [],
            })

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }

    n_img = len(images)
    n_ann = len(annotations)
    n_cat = len(categories)
    print(f"Kaggle 학습 데이터 변환 완료: 이미지 {n_img}장, 어노테이션 {n_ann}개, 클래스 {n_cat}개")
    return coco
