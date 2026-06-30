"""COCO 포맷 → YOLO 포맷 변환."""

import json
import os
from collections import defaultdict
from pathlib import Path

import yaml


def coco_to_yolo(
    coco: dict,
    output_dir: str | Path,
    split: str = "train",
    symlink: bool = True,
) -> Path:
    """COCO dict를 YOLO 디렉토리 구조로 변환.

    Args:
        coco: COCO format dict
        output_dir: YOLO 데이터셋 루트 경로
        split: "train" 또는 "val"
        symlink: True면 이미지 심볼릭 링크, False면 복사

    Returns:
        output_dir Path
    """
    output_dir = Path(output_dir)
    image_dir = output_dir / "images" / split
    label_dir = output_dir / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    img_id_to_info = {img["id"]: img for img in coco["images"]}

    anns_by_img = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_img[ann["image_id"]].append(ann)

    cat_id_to_yolo_idx = {}
    for i, cat in enumerate(coco["categories"]):
        cat_id_to_yolo_idx[cat["id"]] = i

    for img_id, img_info in img_id_to_info.items():
        src_path = Path(img_info["file_name"])
        dst_name = src_path.name
        dst_path = image_dir / dst_name

        if not dst_path.exists():
            if symlink:
                os.symlink(src_path.resolve(), dst_path)
            else:
                import shutil
                shutil.copy2(src_path, dst_path)

        img_w = img_info["width"]
        img_h = img_info["height"]
        txt_name = src_path.stem + ".txt"

        with open(label_dir / txt_name, "w") as f:
            for ann in anns_by_img.get(img_id, []):
                cls_idx = cat_id_to_yolo_idx[ann["category_id"]]
                x_min, y_min, w, h = ann["bbox"]
                x_center = (x_min + w / 2) / img_w
                y_center = (y_min + h / 2) / img_h
                norm_w = w / img_w
                norm_h = h / img_h
                f.write(f"{cls_idx} {x_center:.6f} {y_center:.6f} "
                        f"{norm_w:.6f} {norm_h:.6f}\n")

    return output_dir


def write_yolo_yaml(
    coco: dict,
    output_dir: str | Path,
    yaml_name: str = "data.yaml",
) -> Path:
    """YOLO data.yaml 및 class_map.json 생성.

    class_map.json: {yolo_idx → original_category_id} 역매핑 저장.
    추론 시 YOLO 예측 클래스를 원본 category_id로 변환하는 데 사용.
    """
    output_dir = Path(output_dir)
    names = {
        i: cat["name"] for i, cat in enumerate(coco["categories"])
    }

    data_yaml = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": len(names),
        "names": names,
    }

    yaml_path = output_dir / yaml_name
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data_yaml, f, allow_unicode=True, default_flow_style=False)

    # yolo_idx → original category_id 매핑 저장
    class_map = {
        i: {"category_id": cat["id"], "name": cat["name"]}
        for i, cat in enumerate(coco["categories"])
    }
    class_map_path = output_dir / "class_map.json"
    with open(class_map_path, "w", encoding="utf-8") as f:
        json.dump(class_map, f, ensure_ascii=False, indent=2)

    print(f"YOLO yaml 저장: {yaml_path}")
    print(f"클래스 매핑 저장: {class_map_path}")
    return yaml_path


def prepare_yolo_dataset(
    train_coco: dict,
    val_coco: dict,
    output_dir: str | Path,
    symlink: bool = True,
) -> Path:
    """train/val COCO를 한번에 YOLO 데이터셋으로 변환."""
    output_dir = Path(output_dir)

    coco_to_yolo(train_coco, output_dir, split="train", symlink=symlink)
    coco_to_yolo(val_coco, output_dir, split="val", symlink=symlink)

    yaml_path = write_yolo_yaml(train_coco, output_dir)

    train_imgs = len(train_coco["images"])
    val_imgs = len(val_coco["images"])
    print(f"YOLO 데이터셋 준비 완료: train {train_imgs}장, val {val_imgs}장")

    return yaml_path
