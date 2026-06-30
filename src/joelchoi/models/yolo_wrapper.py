"""ultralytics YOLO를 실험 config로 실행하는 래퍼."""

from pathlib import Path

from ultralytics import YOLO


YOLO_MODELS = {
    "yolo11n": "yolo11n.pt",
    "yolo11s": "yolo11s.pt",
    "yolo11m": "yolo11m.pt",
    "yolo11l": "yolo11l.pt",
    "yolo11x": "yolo11x.pt",
}


def train_yolo(
    config: dict,
    data_yaml: str | Path,
    project_dir: str | Path,
) -> dict:
    """config 기반으로 YOLO 학습 실행.

    Args:
        config: 실험 config dict
        data_yaml: YOLO data.yaml 경로
        project_dir: 결과 저장 디렉토리

    Returns:
        학습 결과 metrics dict
    """
    model_cfg = config["model"]
    train_cfg = config["training"]
    data_cfg = config.get("data", {})
    exp_name = config["experiment"]["name"]

    model_name = model_cfg["name"]
    weights = YOLO_MODELS.get(model_name, f"{model_name}.pt")
    model = YOLO(weights)

    train_args = {
        "data": str(data_yaml),
        "epochs": train_cfg.get("epochs", 50),
        "imgsz": data_cfg.get("img_size", 640),
        "batch": train_cfg.get("batch_size", 16),
        "project": str(project_dir),
        "name": exp_name,
        "exist_ok": True,
    }

    if "optimizer" in train_cfg:
        train_args["optimizer"] = train_cfg["optimizer"]
    if "lr" in train_cfg:
        train_args["lr0"] = train_cfg["lr"]
    if "weight_decay" in train_cfg:
        train_args["weight_decay"] = train_cfg["weight_decay"]
    if "patience" in train_cfg:
        train_args["patience"] = train_cfg["patience"]

    YOLO_AUG_KEYS = [
        "mosaic",
        "mixup",
        "copy_paste",
        "hsv_h",
        "hsv_s",
        "hsv_v",
        "degrees",
        "translate",
        "scale",
        "shear",
        "perspective",
        "fliplr",
        "flipud",
        "erasing",
        "auto_augment",
        "close_mosaic",
    ]
    aug_cfg = config.get("augmentation", {})
    for key in YOLO_AUG_KEYS:
        if key in aug_cfg:
            train_args[key] = aug_cfg[key]

    results = model.train(**train_args)

    return results


def evaluate_yolo(
    weights_path: str | Path,
    data_yaml: str | Path,
) -> dict:
    """YOLO 모델 평가."""
    model = YOLO(str(weights_path))
    metrics = model.val(data=str(data_yaml))

    return {
        "map75": float(metrics.box.map75),
        "map75_95": float(metrics.box.map),
    }
