"""기존 YOLO 포맷 데이터셋 → COCO dict 변환 (torchvision 모델에서 사용)."""

from pathlib import Path

import yaml
from PIL import Image


def load_yolo_as_coco(
    yolo_dir: str | Path,
    split: str = "train",
) -> dict:
    """YOLO 디렉토리를 읽어 COCO format dict로 반환.

    Args:
        yolo_dir: YOLO 데이터셋 루트 (data.yaml이 있는 디렉토리)
        split: "train" 또는 "val"

    Returns:
        COCO format dict
    """
    yolo_dir = Path(yolo_dir)
    yaml_path = yolo_dir / "data.yaml"

    with open(yaml_path, encoding="utf-8") as f:
        data_yaml = yaml.safe_load(f)

    names = data_yaml["names"]
    if isinstance(names, dict):
        categories = [
            {"id": int(k) + 1, "name": v, "supercategory": "pill"}
            for k, v in sorted(names.items(), key=lambda x: int(x[0]))
        ]
        idx_to_cat_id = {int(k): int(k) + 1 for k in names}
    else:
        categories = [
            {"id": i + 1, "name": name, "supercategory": "pill"}
            for i, name in enumerate(names)
        ]
        idx_to_cat_id = {i: i + 1 for i in range(len(names))}

    image_dir = yolo_dir / "images" / split
    label_dir = yolo_dir / "labels" / split

    images = []
    annotations = []
    img_id = 0
    ann_id = 0

    for img_path in sorted(image_dir.glob("*")):
        if img_path.suffix.lower() not in (".png", ".jpg", ".jpeg"):
            continue

        img = Image.open(img_path)
        img_w, img_h = img.size
        img.close()

        img_id += 1
        images.append({
            "id": img_id,
            "file_name": str(img_path.resolve()),
            "width": img_w,
            "height": img_h,
        })

        label_path = label_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            continue

        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                cls_idx = int(parts[0])
                xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

                x_min = (xc - w / 2) * img_w
                y_min = (yc - h / 2) * img_h
                box_w = w * img_w
                box_h = h * img_h

                ann_id += 1
                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": idx_to_cat_id[cls_idx],
                    "bbox": [x_min, y_min, box_w, box_h],
                    "area": box_w * box_h,
                    "iscrowd": 0,
                    "segmentation": [],
                })

    print(f"[{split}] 이미지: {len(images)}장, 어노테이션: {len(annotations)}개, "
          f"클래스: {len(categories)}개")

    return {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
