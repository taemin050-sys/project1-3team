"""COCO 데이터셋에서 서브셋을 생성.

중요(누수 방지):
AI Hub / Kaggle 경구약제 데이터는 "같은 약 조합"을 여러 각도/회전으로 촬영합니다.
파일명 접두사(예: ``K-001900-016548-019607-029451``)가 같은 이미지는 사실상
동일 장면이므로, train/val을 이미지 단위로 랜덤 분할하면 거의 같은 사진이
양쪽에 섞여 들어가 검증 점수가 비현실적으로 부풀려집니다.
따라서 이 모듈은 **조합(combo) 접두사 단위(group)** 로 분할합니다.
"""

import json
import random
import re
from collections import defaultdict
from pathlib import Path

from sklearn.model_selection import GroupShuffleSplit

TIER_SIZES = {
    "tiny": 100,
    "small": 500,
    "medium": 2000,
    "large": 6000,
}


def combo_group_key(file_name: str) -> str:
    """이미지 파일명 → 약 조합 그룹 키.

    예) ".../K-001900-016548-019607-029451_0_2_0_2_75_000_200.png"
        → "K-001900-016548-019607-029451"
    접두사를 못 찾으면 파일명 stem 전체를 그룹 키로 사용(안전한 fallback).
    """
    stem = Path(file_name).stem
    m = re.match(r"(K-[0-9\-]+?)(?=_)", stem)
    if m:
        return m.group(1)
    return stem.split("_")[0] if "_" in stem else stem


def create_subset(
    coco: dict,
    tier: str = "small",
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[dict, dict]:
    """COCO 데이터셋에서 tier 크기만큼 서브셋을 만들고 train/val로 **그룹 분할**.

    - 같은 조합(combo) 접두사 이미지는 통째로 train 또는 val 한쪽에만 들어갑니다.
    - tier 크기 제한도 조합 단위로 적용해 한 조합이 쪼개지지 않게 합니다.

    Args:
        coco: COCO format dict (images, annotations, categories)
        tier: "tiny", "small", "medium", "large", "full"
        test_size: val 비율(그룹 기준 근사)
        seed: 랜덤 시드

    Returns:
        (train_coco, val_coco) 튜플
    """
    images = coco["images"]
    annotations = coco["annotations"]
    categories = coco["categories"]

    # 이미지별 그룹 키
    groups_by_img = {img["id"]: combo_group_key(img["file_name"]) for img in images}

    # ── 1) tier 크기 제한: 조합(group) 단위로 무작위 선택 ─────────────────
    if tier != "full" and tier in TIER_SIZES:
        max_images = TIER_SIZES[tier]
        if len(images) > max_images:
            g2imgs = defaultdict(list)
            for img in images:
                g2imgs[groups_by_img[img["id"]]].append(img)

            gkeys = list(g2imgs)
            random.Random(seed).shuffle(gkeys)

            kept = []
            for g in gkeys:
                if len(kept) >= max_images:
                    break
                kept.extend(g2imgs[g])  # 조합은 통째로 추가(쪼개지 않음)
            images = kept

    # ── 2) train/val 그룹 분할 ────────────────────────────────────────────
    groups = [groups_by_img[img["id"]] for img in images]
    n_groups = len(set(groups))

    if n_groups < 2:
        # 그룹이 1개뿐이면 분할 불가 → 전부 train, val 비움
        train_imgs, val_imgs = images, []
    else:
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_idx, val_idx = next(gss.split(images, groups=groups))
        train_imgs = [images[i] for i in train_idx]
        val_imgs = [images[i] for i in val_idx]

    train_img_ids = {img["id"] for img in train_imgs}
    val_img_ids = {img["id"] for img in val_imgs}

    train_coco = {
        "images": train_imgs,
        "annotations": [a for a in annotations if a["image_id"] in train_img_ids],
        "categories": categories,
    }
    val_coco = {
        "images": val_imgs,
        "annotations": [a for a in annotations if a["image_id"] in val_img_ids],
        "categories": categories,
    }

    # ── 3) 누수 검증(같은 조합이 양쪽에 있으면 즉시 실패) ─────────────────
    tr_groups = {groups_by_img[i] for i in train_img_ids}
    va_groups = {groups_by_img[i] for i in val_img_ids}
    overlap = tr_groups & va_groups
    assert not overlap, (
        f"누수 발생: train/val 공유 조합 {len(overlap)}개 (예: {list(overlap)[:3]})"
    )

    print(
        f"[{tier}] train: {len(train_coco['images'])}장({len(tr_groups)}조합), "
        f"val: {len(val_coco['images'])}장({len(va_groups)}조합) | 조합 누수 0"
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
