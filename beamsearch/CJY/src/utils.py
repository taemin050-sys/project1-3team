import os
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


def load_hf_token(env_key: str = "HG_TOKEN") -> str | None:
    """CJY 프로젝트 루트의 ``.env``에서 HuggingFace 토큰을 읽어, huggingface_hub가
    (transformers의 ``from_pretrained`` 등에서) 자동으로 사용하는 ``HF_TOKEN``
    환경변수로 등록한다.

    이미 ``HF_TOKEN``이 설정돼 있으면(다른 방식으로 이미 로그인된 경우) 건드리지
    않는다. ``.env``가 없거나 키가 없으면 조용히 아무 것도 하지 않는다 — RF-DETR
    베이스 체크포인트처럼 공개 모델은 토큰 없이도 동작하므로 이 함수는 있으면
    쓰고 없으면 넘어가는 선택적 보강용이다(주로 HF Hub의 비로그인 요청 rate
    limit을 피하거나, 나중에 gated 체크포인트로 바꿀 때를 대비한다).
    """
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]

    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return None

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key == env_key and value:
            os.environ["HF_TOKEN"] = value
            return value
    return None
