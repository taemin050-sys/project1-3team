"""학습 데이터 증강용 COCO 병합.

전략:
- Kaggle 제공 train을 **그룹 분할**해 train_base / val_base 를 만든다(val은 테스트와
  같은 분포의 '대표 검증셋'으로 유지).
- AIHub 조합(1,3,4,5,6 등)을 **추가 학습 데이터로만** 합친다(val에는 넣지 않음).

안전장치:
1. 누수 방지: AIHub 이미지의 조합 prefix가 val_base 조합에 있으면 제외.
2. 중복 방지: 동일 파일명(basename)이 이미 base에 있으면 제외.
3. 라벨 무결성: AIHub 혼합 이미지의 박스는 **지우지 않고 모두 유지**한다
   (지우면 라벨 없는 알약이 생겨 학습을 망친다). 56 클래스가 하나도 없는
   '완전 무관' 이미지는 기본적으로 제외(drop_pure_irrelevant).
4. 카테고리 정합성: 반환되는 train/val은 **동일한 categories 리스트**를 공유한다
   (coco_to_yolo가 split별로 인덱스를 매기므로 순서가 같아야 함).
"""

from collections import defaultdict
from pathlib import Path

from .subset import combo_group_key


def merge_for_augmentation(
    base_train: dict,
    base_val: dict,
    extra_coco: dict,
    allowed_ids: set[int] | None = None,
    drop_pure_irrelevant: bool = True,
) -> tuple[dict, dict]:
    """base_train에 extra_coco를 추가 병합. val은 그대로 유지.

    Args:
        base_train: 기준 학습 COCO (예: Kaggle train split)
        base_val:   기준 검증 COCO (예: Kaggle val split) — 변경 없음, 누수 기준
        extra_coco: 추가할 COCO (예: AIHub)
        allowed_ids: 제출 대상 category_id 집합(56). drop_pure_irrelevant에 사용.
        drop_pure_irrelevant: True면 allowed_ids 박스가 하나도 없는 이미지 제외.

    Returns:
        (train_merged, val_merged) — 동일 categories 공유
    """
    val_groups = {combo_group_key(i["file_name"]) for i in base_val["images"]}
    base_names = {Path(i["file_name"]).name for i in base_train["images"]}
    base_names |= {Path(i["file_name"]).name for i in base_val["images"]}

    extra_anns = defaultdict(list)
    for a in extra_coco["annotations"]:
        extra_anns[a["image_id"]].append(a)

    next_img_id = max(i["id"] for i in base_train["images"]) + 1
    next_ann_id = max(a["id"] for a in base_train["annotations"]) + 1

    add_imgs, add_anns = [], []
    n_leak = n_dup = n_irrelevant = 0

    for img in extra_coco["images"]:
        if combo_group_key(img["file_name"]) in val_groups:
            n_leak += 1
            continue
        if Path(img["file_name"]).name in base_names:
            n_dup += 1
            continue
        anns = extra_anns.get(img["id"], [])
        if not anns:
            continue
        if drop_pure_irrelevant and allowed_ids is not None:
            if not any(a["category_id"] in allowed_ids for a in anns):
                n_irrelevant += 1
                continue

        new_img = dict(img)
        new_img["id"] = next_img_id
        add_imgs.append(new_img)
        for a in anns:
            na = dict(a)
            na["id"] = next_ann_id
            na["image_id"] = next_img_id
            na.setdefault("iscrowd", 0)
            na.setdefault("segmentation", [])
            add_anns.append(na)
            next_ann_id += 1
        next_img_id += 1

    # ── 통합 categories: '실제로 등장하는' category_id만 포함 ──────────────
    # (드롭된 이미지의 클래스가 categories에 남지 않도록 annotation 기준으로 구성)
    name_lookup: dict[int, str] = {}
    for c in (
        base_train["categories"]
        + base_val["categories"]
        + extra_coco["categories"]
    ):
        name_lookup.setdefault(c["id"], c.get("name", str(c["id"])))

    present_ids = {a["category_id"] for a in base_train["annotations"] + add_anns}
    present_ids |= {a["category_id"] for a in base_val["annotations"]}
    categories = [
        {"id": cid, "name": name_lookup.get(cid, str(cid))}
        for cid in sorted(present_ids)
    ]

    train_merged = {
        "images": base_train["images"] + add_imgs,
        "annotations": base_train["annotations"] + add_anns,
        "categories": categories,
    }
    val_merged = {
        "images": base_val["images"],
        "annotations": base_val["annotations"],
        "categories": categories,  # train과 동일 리스트
    }

    # ── 누수 재검증 ───────────────────────────────────────────────────────
    tr_groups = {combo_group_key(i["file_name"]) for i in train_merged["images"]}
    leak = tr_groups & val_groups
    assert not leak, f"누수: train/val 공유 조합 {len(leak)}개"

    print(
        f"증강 병합 완료: base train {len(base_train['images'])} + AIHub {len(add_imgs)} "
        f"= {len(train_merged['images'])}장 | val {len(val_merged['images'])}장(유지)"
    )
    print(
        f"  제외: 누수 {n_leak}, 중복 {n_dup}, 무관(56없음) {n_irrelevant} | "
        f"클래스 {len(categories)}개"
    )
    return train_merged, val_merged
