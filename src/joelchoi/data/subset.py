"""COCO 데이터셋에서 서브셋을 생성."""

import json
from collections import defaultdict
from pathlib import Path

from sklearn.model_selection import train_test_split

TIER_SIZES = {
    "tiny": 100,
    "small": 500,
    "medium": 2000,
    "large": 6000,
}


def create_subset(
    coco: dict,
    tier: str = "small",
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[dict, dict]:
    """COCO 데이터셋에서 tier 크기만큼 서브셋을 생성하고 train/val로 분할.

    Args:
        coco: COCO format dict (images, annotations, categories)
        tier: "tiny", "small", "medium", "large", "full"
        test_size: val 비율
        seed: 랜덤 시드

    Returns:
        (train_coco, val_coco) 튜플
    """
    images = coco["images"]
    annotations = coco["annotations"]
    categories = coco["categories"]

    img_to_anns = defaultdict(list)
    for ann in annotations:
        img_to_anns[ann["image_id"]].append(ann)

    img_to_cats = {}
    for img in images:
        cats = {a["category_id"] for a in img_to_anns.get(img["id"], [])}
        img_to_cats[img["id"]] = min(cats) if cats else 0

    def _safe_stratify(labels):
        from collections import Counter

        counts = Counter(labels)
        if min(counts.values()) >= 2:
            return labels
        return None

    if tier != "full" and tier in TIER_SIZES:
        max_images = TIER_SIZES[tier]
        if len(images) > max_images:
            cat_labels = [img_to_cats[img["id"]] for img in images]
            _, keep_idx = train_test_split(
                range(len(images)),
                test_size=max_images / len(images),
                stratify=_safe_stratify(cat_labels),
                random_state=seed,
            )
            images = [images[i] for i in sorted(keep_idx)]

    cat_labels = [img_to_cats[img["id"]] for img in images]
    indices = list(range(len(images)))

    train_idx, val_idx = train_test_split(
        indices,
        test_size=test_size,
        stratify=_safe_stratify(cat_labels),
        random_state=seed,
    )

    train_img_ids = {images[i]["id"] for i in train_idx}
    val_img_ids = {images[i]["id"] for i in val_idx}

    train_coco = {
        "images": [images[i] for i in sorted(train_idx)],
        "annotations": [a for a in annotations if a["image_id"] in train_img_ids],
        "categories": categories,
    }
    val_coco = {
        "images": [images[i] for i in sorted(val_idx)],
        "annotations": [a for a in annotations if a["image_id"] in val_img_ids],
        "categories": categories,
    }

    print(
        f"[{tier}] train: {len(train_coco['images'])}장, "
        f"val: {len(val_coco['images'])}장"
    )

    return train_coco, val_coco


def save_split(
    train_coco: dict,
    val_coco: dict,
    output_dir: str | Path,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, data in [("train", train_coco), ("val", val_coco)]:
        path = output_dir / f"{name}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        print(f"저장: {path}")
