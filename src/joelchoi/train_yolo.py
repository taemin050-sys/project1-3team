"""YOLO 학습을 실험 config로 실행하는 진입점."""

import json
from pathlib import Path

from src.joelchoi.models.yolo_wrapper import evaluate_yolo, train_yolo
from src.joelchoi.utils import save_config, set_seed


def run_yolo_experiment(
    config: dict,
    data_yaml: str | Path,
    project_dir: str | Path,
) -> dict:
    """config 기반 YOLO 실험 실행.

    Args:
        config: 실험 config dict
        data_yaml: YOLO data.yaml 경로
        project_dir: 결과 저장 디렉토리

    Returns:
        metrics dict
    """
    exp_name = config["experiment"]["name"]
    seed = config["experiment"].get("seed", 42)
    set_seed(seed)

    output_dir = Path(project_dir) / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, output_dir / "config.yaml")

    train_yolo(config, data_yaml, project_dir)

    best_weights = output_dir / "weights" / "best.pt"
    if best_weights.exists():
        metrics = evaluate_yolo(best_weights, data_yaml)
    else:
        metrics = {"map75": 0.0, "map75_95": 0.0}

    final_metrics = {
        **metrics,
        "model": config["model"]["name"],
        "epochs": config["training"].get("epochs", 50),
    }

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2)
    print(f"\n결과 저장: {output_dir}")

    return final_metrics
