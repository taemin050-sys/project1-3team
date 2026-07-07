"""학습된 YOLO 모델로 테스트 이미지 추론 → Kaggle 제출 JSON 생성."""

import json
from pathlib import Path

from ultralytics import YOLO


KAGGLE_TEST_DIR = (
    Path.home()
    / ".cache/kagglehub/competitions/ai12-level1-project/sprint_ai_project1_data"
    / "test_images"
)


def load_class_map(class_map_path: str | Path) -> dict[int, int]:
    """class_map.json 로드 → {yolo_idx: category_id} 반환."""
    with open(class_map_path, encoding="utf-8") as f:
        raw = json.load(f)
    # JSON 키는 문자열이므로 int 변환
    return {int(k): v["category_id"] for k, v in raw.items()}


def restrict_class_map(
    class_map_path: str | Path,
    allowed_ids,
    output_path: str | Path | None = None,
) -> Path:
    """제출 대상 category_id만 남기고 나머지는 -1로 표시.

    증강 학습 시 모델은 56개 외 클래스(AIHub 전용 약)도 함께 배운다. 추론은
    category_id == -1 예측을 자동으로 버리므로, 제출 전 class_map에서 56개
    이외 클래스를 -1로 바꿔 두면 제출에 무관한 클래스가 섞이지 않는다.

    Args:
        class_map_path: 원본 class_map.json (모든 클래스, K-코드 id)
        allowed_ids: 제출 허용 category_id 집합(예: Kaggle 56)
        output_path: 저장 경로(None이면 ``*_submit.json``)

    Returns:
        저장된 class_map 경로
    """
    allowed = set(int(x) for x in allowed_ids)
    class_map_path = Path(class_map_path)
    with open(class_map_path, encoding="utf-8") as f:
        cm = json.load(f)

    kept = 0
    for info in cm.values():
        if info["category_id"] in allowed:
            kept += 1
        else:
            info["category_id"] = -1

    if output_path is None:
        output_path = class_map_path.with_name(class_map_path.stem + "_submit.json")
    output_path = Path(output_path)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cm, f, ensure_ascii=False, indent=2)
    print(f"제출용 class_map: {kept}/{len(cm)}개 클래스 유지(나머지 -1) → {output_path}")
    return output_path


def build_class_map_from_yaml(
    yaml_path: str | Path,
    kaggle_ann_root: str | Path | None = None,
) -> dict:
    """기존 data.yaml에서 class_map.json을 역으로 생성.

    AI Hub 데이터로 학습한 모델처럼 class_map.json이 없는 경우,
    data.yaml의 drug 이름과 Kaggle 어노테이션의 category_id를 이름으로 매핑합니다.
    Kaggle 56 클래스에 없는 AI Hub 전용 클래스는 category_id=-1로 표시됩니다.

    Args:
        yaml_path: data.yaml 경로
        kaggle_ann_root: Kaggle train_annotations 경로 (None이면 기본 경로)

    Returns:
        class_map dict ({yolo_idx: {category_id, name}})
    """
    import yaml as _yaml

    yaml_path = Path(yaml_path)
    with open(yaml_path, encoding="utf-8") as f:
        data = _yaml.safe_load(f)
    yolo_names: dict[int, str] = {int(k): v for k, v in data["names"].items()}

    if kaggle_ann_root is None:
        kaggle_ann_root = (
            Path.home()
            / ".cache/kagglehub/competitions/ai12-level1-project/sprint_ai_project1_data"
            / "train_annotations"
        )
    kaggle_ann_root = Path(kaggle_ann_root)

    # 폴더명 K-XXXXXX의 숫자 부분이 category_id (K-001900 → 1900)
    kaggle_name_to_id: dict[str, int] = {}
    for f in kaggle_ann_root.rglob("*.json"):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            for cat in d.get("categories", []):
                kaggle_name_to_id[cat["name"].strip()] = cat["id"]
        except Exception:
            continue

    class_map = {}
    for idx, name in yolo_names.items():
        cat_id = kaggle_name_to_id.get(name.strip(), -1)
        class_map[idx] = {"category_id": cat_id, "name": name}

    matched = sum(1 for v in class_map.values() if v["category_id"] != -1)
    print(f"클래스 매핑: {matched}/{len(class_map)}개 Kaggle category_id 매칭")

    output_path = yaml_path.parent / "class_map.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(class_map, f, ensure_ascii=False, indent=2)
    print(f"class_map.json 저장: {output_path}")
    return class_map


def run_inference(
    weights_path: str | Path,
    class_map_path: str | Path,
    test_dir: str | Path | None = None,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    img_size: int = 640,
    batch_size: int = 32,
) -> list[dict]:
    """테스트 이미지 전체에 추론 실행.

    Args:
        weights_path: 학습된 모델 가중치 경로 (best.pt)
        class_map_path: class_map.json 경로
        test_dir: 테스트 이미지 폴더 (None이면 Kaggle 기본 경로)
        conf_threshold: confidence 임계값
        iou_threshold: NMS IoU 임계값
        img_size: 추론 이미지 크기
        batch_size: 배치 크기

    Returns:
        COCO predictions list:
        [{"image_id": int, "category_id": int, "bbox": [...], "score": float}, ...]
    """
    test_dir = Path(test_dir) if test_dir else KAGGLE_TEST_DIR
    yolo_to_cat = load_class_map(class_map_path)

    model = YOLO(str(weights_path))

    test_images = sorted(test_dir.glob("*.png"), key=lambda p: int(p.stem))
    print(f"추론 대상: {len(test_images)}장")

    predictions = []
    results_iter = model.predict(
        source=[str(p) for p in test_images],
        conf=conf_threshold,
        iou=iou_threshold,
        imgsz=img_size,
        batch=batch_size,
        verbose=False,
        stream=True,
    )

    for img_path, result in zip(test_images, results_iter):
        image_id = int(img_path.stem)
        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            continue

        for box in boxes:
            cls_idx = int(box.cls.item())
            score = float(box.conf.item())
            # xyxy → xywh (COCO 형식)
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            bbox = [x1, y1, x2 - x1, y2 - y1]

            cat_id = yolo_to_cat.get(cls_idx, -1)
            if cat_id == -1:
                continue
            predictions.append({
                "image_id": image_id,
                "category_id": cat_id,
                "bbox": [round(v, 2) for v in bbox],
                "score": round(score, 6),
            })

    print(
        f"총 예측 수: {len(predictions)}개 (이미지당 평균 {len(predictions) / len(test_images):.1f}개)"
    )
    return predictions


def save_submission(
    predictions: list[dict],
    output_path: str | Path,
) -> Path:
    """예측 결과를 CSV 제출 파일로 저장.

    컬럼: annotation_id, image_id, category_id, bbox_x, bbox_y, bbox_w, bbox_h, score
    """
    import csv

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "annotation_id",
            "image_id",
            "category_id",
            "bbox_x",
            "bbox_y",
            "bbox_w",
            "bbox_h",
            "score",
        ])
        for ann_id, pred in enumerate(predictions, start=1):
            x, y, w, h = pred["bbox"]
            writer.writerow([
                ann_id,
                pred["image_id"],
                pred["category_id"],
                int(x),
                int(y),
                int(w),
                int(h),
                float(int(pred["score"] * 100) / 100),
            ])

    print(f"제출 파일 저장: {output_path} ({len(predictions)}개 예측)")
    return output_path
