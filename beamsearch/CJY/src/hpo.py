"""Optuna 기반 하이퍼파라미터 탐색.

기존 학습 함수(`train_yolo::run_yolo_experiment` / `train::train_torchvision` /
`train_hf::train_rfdetr`)를 그대로 재사용하면서, config의 일부 값을 trial마다
Optuna가 제안하는 값으로 바꿔가며 반복 학습 → mAP@75:95를 목적함수로 최적화한다.

설계 원칙:
- 전체 epoch을 다 돌리며 탐색하면 너무 비싸므로, ``hpo_epochs``로 trial마다
  epoch 수를 줄여 "좋은 방향"만 빠르게 찾는 용도로 쓰는 걸 권장한다. 최종
  후보는 원래 config의 전체 epoch으로 별도 재학습해서 확정할 것.
- 각 trial은 실험 이름에 ``_trial{N}``을 붙여 다른 실험들과 동일한
  ``experiments/<name>/metrics.json`` 구조로 저장되므로, 기존 비교 테이블
  코드를 그대로 재사용할 수 있다.
- 모델 계열(YOLO/torchvision/HuggingFace)마다 학습 함수 시그니처가 달라
  ``model.type``으로 분기한다(`run_experiment_augmented.ipynb`의 모델 비교
  루프와 동일 원칙).
- pruning(중간 성능이 나쁜 trial 조기 종료)은 아직 지원하지 않는다 — 세
  학습 함수 모두 epoch마다 Optuna에 중간값을 보고하도록 고치는 추가 작업이
  필요해서 1차 버전에서는 뺐다. 필요하면 다음 확장 포인트로 남겨둔다.
"""

import copy
from pathlib import Path
from typing import Any

import optuna

# model.type별 기본 탐색 공간. "dotted.key" -> (종류, ...범위/선택지)
DEFAULT_SEARCH_SPACES: dict[str, dict[str, tuple]] = {
    "yolo": {
        "training.lr": ("loguniform", 1e-4, 5e-2),
        "training.weight_decay": ("loguniform", 1e-5, 1e-2),
        "training.batch_size": ("categorical", [8, 16, 32]),
        "training.optimizer": ("categorical", ["SGD", "AdamW"]),
    },
    "torchvision": {
        "training.lr": ("loguniform", 5e-4, 2e-2),
        "training.weight_decay": ("loguniform", 1e-5, 1e-2),
        "training.batch_size": ("categorical", [2, 4, 8]),
    },
    "huggingface": {
        "training.lr": ("loguniform", 1e-5, 5e-4),
        "training.weight_decay": ("loguniform", 1e-6, 1e-3),
        "training.batch_size": ("categorical", [2, 4, 8]),
    },
}


def _set_nested(d: dict, dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    node = d
    for k in keys[:-1]:
        node = node[k]
    node[keys[-1]] = value


def _suggest(trial: optuna.Trial, name: str, spec: tuple) -> Any:
    kind = spec[0]
    if kind == "loguniform":
        _, low, high = spec
        return trial.suggest_float(name, low, high, log=True)
    if kind == "uniform":
        _, low, high = spec
        return trial.suggest_float(name, low, high)
    if kind == "int":
        _, low, high = spec
        return trial.suggest_int(name, low, high)
    if kind == "categorical":
        _, choices = spec
        return trial.suggest_categorical(name, choices)
    raise ValueError(f"지원하지 않는 탐색 종류: {kind}")


def build_trial_config(
    base_config: dict,
    trial: optuna.Trial,
    search_space: dict[str, tuple],
    hpo_epochs: int | None,
    trial_tag: str,
) -> dict:
    """base_config를 복사해 trial이 제안한 값으로 덮어쓴 새 config를 만든다."""
    cfg = copy.deepcopy(base_config)
    for dotted_key, spec in search_space.items():
        value = _suggest(trial, dotted_key, spec)
        _set_nested(cfg, dotted_key, value)

    if hpo_epochs is not None:
        cfg["training"]["epochs"] = hpo_epochs

    cfg["experiment"]["name"] = f"{base_config['experiment']['name']}_{trial_tag}"
    return cfg


def run_hpo(
    base_config: dict,
    project_dir: str | Path,
    train_coco: dict | None = None,
    val_coco: dict | None = None,
    data_yaml: str | Path | None = None,
    search_space: dict[str, tuple] | None = None,
    n_trials: int = 20,
    hpo_epochs: int | None = None,
    seed: int = 42,
    study_name: str | None = None,
    storage: str | None = None,
    direction: str = "maximize",
) -> optuna.Study:
    """base_config 기준으로 Optuna 베이지안 탐색을 수행.

    Args:
        base_config: 기준 실험 config(exp011~017 등에서 `load_config`로 읽은 dict).
        project_dir: 실험 결과 저장 경로(다른 실험들과 같은 `experiments/`).
        train_coco, val_coco: torchvision/huggingface(model.type)용 COCO dict.
        data_yaml: YOLO(model.type == "yolo")용 data.yaml 경로. 하이퍼파라미터
            탐색은 보통 데이터를 바꾸지 않으므로 trial마다 새로 만들지 않고
            미리 한 번 만든 경로를 그대로 재사용한다.
        search_space: ``{"training.lr": ("loguniform", lo, hi), ...}``. None이면
            ``DEFAULT_SEARCH_SPACES[base_config["model"]["type"]]``를 사용.
        n_trials: 시도 횟수.
        hpo_epochs: trial마다 쓸 epoch 수(None이면 base_config의 epoch 그대로 —
            보통 이러면 너무 오래 걸리므로 낮은 값을 권장).
        seed: TPESampler 시드.
        study_name, storage: Optuna Study 식별자/저장소(예: sqlite로 재개 가능하게).
        direction: "maximize"(mAP 기준이므로 기본값 유지 권장).

    Returns:
        optuna.Study. `study.best_params`, `study.best_value`,
        `study.best_trial.user_attrs["metrics"]`로 결과 확인.
    """
    model_type = base_config["model"]["type"]
    search_space = search_space or DEFAULT_SEARCH_SPACES.get(model_type)
    if not search_space:
        raise ValueError(
            f"model.type={model_type}에 대한 기본 탐색 공간이 없습니다. search_space를 직접 지정하세요."
        )
    if model_type == "yolo" and data_yaml is None:
        raise ValueError("model.type == 'yolo'는 data_yaml이 필요합니다.")
    if model_type in ("torchvision", "huggingface") and (train_coco is None or val_coco is None):
        raise ValueError(f"model.type == '{model_type}'는 train_coco/val_coco가 필요합니다.")

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(
        direction=direction,
        sampler=sampler,
        study_name=study_name or f"{base_config['experiment']['name']}_hpo",
        storage=storage,
        load_if_exists=storage is not None,
    )

    def objective(trial: optuna.Trial) -> float:
        cfg = build_trial_config(base_config, trial, search_space, hpo_epochs, f"trial{trial.number}")
        print(f"\n--- trial {trial.number}: {cfg['experiment']['name']} ---")
        for dotted_key in search_space:
            keys = dotted_key.split(".")
            v = cfg
            for k in keys:
                v = v[k]
            print(f"  {dotted_key} = {v}")

        if model_type == "yolo":
            from src.train_yolo import run_yolo_experiment

            metrics = run_yolo_experiment(cfg, data_yaml, project_dir)
            score = metrics.get("map75_95", 0.0)
        elif model_type == "torchvision":
            from src.train import train_torchvision

            metrics = train_torchvision(cfg, train_coco, val_coco, project_dir)
            score = metrics.get("best_map75_95", 0.0)
        elif model_type == "huggingface":
            from src.train_hf import train_rfdetr

            metrics = train_rfdetr(cfg, train_coco, val_coco, project_dir)
            score = metrics.get("best_map75_95", 0.0)
        else:
            raise ValueError(f"알 수 없는 model.type: {model_type}")

        trial.set_user_attr("exp_name", cfg["experiment"]["name"])
        trial.set_user_attr("metrics", metrics)
        return score

    study.optimize(objective, n_trials=n_trials)

    print(f"\n최적 trial: #{study.best_trial.number} | mAP@75:95={study.best_value:.4f}")
    print("최적 하이퍼파라미터:", study.best_params)

    return study


def summarize_study(study: optuna.Study):
    """완료된 trial들을 mAP 내림차순 DataFrame으로 정리."""
    import pandas as pd

    rows = []
    for t in study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue
        rows.append({"trial": t.number, "mAP@75:95": t.value, **t.params})

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("mAP@75:95", ascending=False)
    return df
