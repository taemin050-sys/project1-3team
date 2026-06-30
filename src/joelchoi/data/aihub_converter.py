"""AI Hub 경구약제 조합 데이터 → COCO 통합 포맷 변환."""

import json
from collections import defaultdict
from pathlib import Path


EXCLUDED_COMBOS = {2}


def find_aihub_root(base_path: str | Path | None = None) -> Path:
    if base_path:
        return Path(base_path)
    default = (
        Path.home()
        / "Downloads"
        / "166.약품식별 인공지능 개발을 위한 경구약제 이미지 데이터"
        / "01.데이터"
        / "1.Training"
    )
    if default.exists():
        return default
    raise FileNotFoundError(
        "AI Hub 데이터 경로를 찾을 수 없습니다. base_path를 지정해주세요."
    )


def _collect_label_dir(root: Path) -> dict[int, Path]:
    label_base = root / "라벨링데이터" / "경구약제조합 5000종"
    result = {}
    for d in sorted(label_base.iterdir()):
        if d.is_dir() and d.name.startswith("TL_"):
            parts = d.name.split("_")
            combo_num = int(parts[1])
            result[combo_num] = d
    return result


def _collect_image_dir(root: Path) -> dict[int, Path]:
    image_base = root / "원천데이터" / "경구약제조합 5000종"
    result = {}
    for d in sorted(image_base.iterdir()):
        if d.is_dir() and d.name.startswith("TS_"):
            parts = d.name.split("_")
            combo_num = int(parts[1])
            result[combo_num] = d
    return result


def convert_combo(
    label_dir: Path,
    image_dir: Path,
    combo_num: int,
    category_map: dict[str, int],
    drug_names: dict[str, str],
    image_id_offset: int = 0,
    ann_id_offset: int = 0,
) -> tuple[list[dict], list[dict], int, int]:
    """단일 조합 세트(TL_X, TS_X)를 COCO images/annotations로 변환."""
    images = []
    annotations = []
    img_id = image_id_offset
    ann_id = ann_id_offset

    combo_dirs = sorted(
        d for d in label_dir.iterdir() if d.is_dir() and d.name.endswith("_json")
    )

    for combo_label_dir in combo_dirs:
        combo_name = combo_label_dir.name.replace("_json", "")
        combo_image_dir = image_dir / combo_name

        if not combo_image_dir.exists():
            continue

        img_anns: dict[str, list[dict]] = defaultdict(list)
        img_meta: dict[str, dict] = {}

        for drug_dir in sorted(combo_label_dir.iterdir()):
            if not drug_dir.is_dir():
                continue
            drug_code = drug_dir.name

            if drug_code not in category_map:
                category_map[drug_code] = len(category_map) + 1

            cat_id = category_map[drug_code]

            for json_file in sorted(drug_dir.glob("*.json")):
                data = None
                # 1단계: utf-8 시도
                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    # 2단계: cp949 시도 (안전 장치 추가)
                    try:
                        with open(
                            json_file, "r", encoding="cp949", errors="replace"
                        ) as f:
                            data = json.load(f)
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        # 3단계: euc-kr 시도
                        try:
                            with open(
                                json_file, "r", encoding="euc-kr", errors="replace"
                            ) as f:
                                data = json.load(f)
                        except Exception:
                            pass

                # 예외 처리: 모든 인코딩이 실패해 data를 읽지 못한 경우 안전하게 패스
                if data is None:
                    print(f"경고: {json_file.name} 파일을 읽을 수 없어 건너뜁니다.")
                    continue
                img_info = data["images"][0]
                file_name = img_info["file_name"]

                if "_index" in file_name:
                    continue

                dl_name = img_info.get("dl_name", "")
                if dl_name and drug_code not in drug_names:
                    drug_names[drug_code] = dl_name

                if file_name not in img_meta:
                    img_meta[file_name] = {
                        "width": img_info["width"],
                        "height": img_info["height"],
                    }

                for ann in data["annotations"]:
                    bbox = ann.get("bbox", [])
                    if not isinstance(bbox, list) or len(bbox) != 4:
                        continue
                    if not all(isinstance(v, (int, float)) for v in bbox):
                        continue

                    img_anns[file_name].append({
                        "bbox": [float(v) for v in bbox],
                        "category_id": cat_id,
                        "area": ann.get("area", bbox[2] * bbox[3]),
                        "iscrowd": ann.get("iscrowd", 0),
                    })

        for file_name in sorted(img_anns.keys()):
            img_id += 1
            meta = img_meta[file_name]
            img_path = combo_image_dir / file_name

            images.append({
                "id": img_id,
                "file_name": str(img_path),
                "width": meta["width"],
                "height": meta["height"],
                "combo_num": combo_num,
            })

            for ann_data in img_anns[file_name]:
                ann_id += 1
                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": ann_data["category_id"],
                    "bbox": ann_data["bbox"],
                    "area": ann_data["area"],
                    "iscrowd": ann_data["iscrowd"],
                    "segmentation": [],
                })

    return images, annotations, img_id, ann_id


def convert_aihub_to_coco(
    base_path: str | Path | None = None,
    combo_nums: list[int] | None = None,
    output_path: str | Path | None = None,
) -> dict:
    """AI Hub 조합 데이터를 COCO 통합 포맷으로 변환.

    Args:
        base_path: AI Hub Training 데이터 루트 경로
        combo_nums: 사용할 조합 번호 리스트 (None이면 TS_02 제외 전체)
        output_path: 저장할 JSON 경로 (None이면 저장하지 않음)

    Returns:
        COCO format dict
    """
    root = find_aihub_root(base_path)
    label_dirs = _collect_label_dir(root)
    image_dirs = _collect_image_dir(root)

    available = set(label_dirs.keys()) & set(image_dirs.keys())

    if combo_nums is None:
        combo_nums = sorted(available - EXCLUDED_COMBOS)
    else:
        for n in combo_nums:
            if n in EXCLUDED_COMBOS:
                raise ValueError(
                    f"TS_{n:02d}/TL_{n:02d}는 Kaggle 테스트셋이므로 사용 불가"
                )
            if n not in available:
                raise FileNotFoundError(
                    f"TS_{n:02d}/TL_{n:02d} 데이터가 존재하지 않습니다. "
                    f"사용 가능: {sorted(available)}"
                )

    category_map: dict[str, int] = {}
    drug_names: dict[str, str] = {}
    all_images = []
    all_annotations = []
    img_id_offset = 0
    ann_id_offset = 0

    for num in combo_nums:
        print(f"변환 중: TS_{num:02d}/TL_{num:02d} ...")
        images, annotations, img_id_offset, ann_id_offset = convert_combo(
            label_dir=label_dirs[num],
            image_dir=image_dirs[num],
            combo_num=num,
            category_map=category_map,
            drug_names=drug_names,
            image_id_offset=img_id_offset,
            ann_id_offset=ann_id_offset,
        )
        all_images.extend(images)
        all_annotations.extend(annotations)
        print(f"  → 이미지 {len(images)}장, 어노테이션 {len(annotations)}개")

    categories = [
        {
            "id": cat_id,
            "name": drug_names.get(drug_code, drug_code),
            "supercategory": "pill",
        }
        for drug_code, cat_id in sorted(category_map.items(), key=lambda x: x[1])
    ]

    coco = {
        "images": all_images,
        "annotations": all_annotations,
        "categories": categories,
    }

    print(
        f"\n총 이미지: {len(all_images)}, 어노테이션: {len(all_annotations)}, "
        f"클래스: {len(categories)}"
    )

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(coco, f, ensure_ascii=False)
        print(f"저장: {output_path}")

    return coco
