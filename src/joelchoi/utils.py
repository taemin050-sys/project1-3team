import random
from pathlib import Path

import numpy as np
import torch
import yaml


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def merge_configs(*configs: dict) -> dict:
    merged = {}
    for cfg in configs:
        for key, value in cfg.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = merge_configs(merged[key], value)
            else:
                merged[key] = value
    return merged


def save_config(config: dict, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
