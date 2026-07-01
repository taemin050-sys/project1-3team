"""torchvision detection 모델 학습 루프."""

import json
import time
from pathlib import Path
from tqdm.auto import tqdm

import torch
from torch.utils.data import DataLoader

from src.joelchoi.data.dataset import (
    PillDetectionDataset,
    build_cat2label,
    collate_fn,
    get_train_augment,
    get_val_augment,
)
from src.joelchoi.evaluate import evaluate_coco
from src.joelchoi.models.builder import build_model
from src.joelchoi.utils import get_device, save_config, set_seed

# 모델 계열별 detection head 파라미터 이름 키워드
# (Faster R-CNN: roi_heads.box_predictor / SSD·RetinaNet·FCOS: head.*_head)
_HEAD_KEYS = ("box_predictor", "classification_head", "regression_head")


def _is_head_param(name: str) -> bool:
    return any(k in name for k in _HEAD_KEYS)


def train_one_epoch(model, optimizer, data_loader, device, max_grad_norm=1.0):
    model.train()
    total_loss = 0.0
    num_batches = 0
    component_totals: dict[str, float] = {}

    for images, targets in tqdm(data_loader):
        images = [img.to(device) for img in images]
        targets = [
            {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in t.items()
            }
            for t in targets
        ]

        loss_dict = model(images, targets)
        losses = sum(loss_dict.values())

        if not torch.isfinite(losses):
            optimizer.zero_grad()
            continue

        optimizer.zero_grad()
        losses.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()

        total_loss += losses.item()
        for k, v in loss_dict.items():
            component_totals[k] = component_totals.get(k, 0.0) + v.item()
        num_batches += 1

    avg_components = {k: v / max(num_batches, 1) for k, v in component_totals.items()}
    return total_loss / max(num_batches, 1), avg_components


def train_torchvision(
    config: dict,
    train_coco: dict,
    val_coco: dict,
    project_dir: str | Path,
) -> dict:
    """config 기반으로 torchvision detection 모델 학습.

    Args:
        config: 실험 config dict
        train_coco: train COCO dict
        val_coco: val COCO dict
        project_dir: 결과 저장 경로

    Returns:
        최종 metrics dict
    """
    exp_name = config["experiment"]["name"]
    seed = config["experiment"].get("seed", 42)
    set_seed(seed)

    device = get_device()
    print(f"Device: {device}")

    train_cfg = config["training"]

    # train/val이 동일한 category_id→라벨 매핑을 쓰도록 train 기준으로 생성
    cat2label = build_cat2label(train_coco)
    num_classes = len(cat2label) + 1  # 배경 포함

    model = build_model(config, num_classes)
    model = model.to(device)

    img_size = config.get("data", {}).get("img_size")
    train_ds = PillDetectionDataset(
        train_coco, augment=get_train_augment(img_size), cat2label=cat2label
    )
    val_ds = PillDetectionDataset(
        val_coco, augment=get_val_augment(img_size), cat2label=cat2label
    )

    use_pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg.get("batch_size", 8),
        shuffle=True,
        num_workers=train_cfg.get("num_workers", 2),
        collate_fn=collate_fn,
        pin_memory=use_pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg.get("batch_size", 8),
        shuffle=False,
        num_workers=train_cfg.get("num_workers", 2),
        collate_fn=collate_fn,
        pin_memory=use_pin_memory,
    )

    optimizer_name = train_cfg.get("optimizer", "SGD")
    lr = train_cfg.get("lr", 0.005)
    head_lr_mult = train_cfg.get("head_lr_mult", 10.0)
    weight_decay = train_cfg.get("weight_decay", 0.0005)
    momentum = train_cfg.get("momentum", 0.9)

    # 새로 초기화된 헤드에는 backbone보다 높은 LR 적용
    head_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad and _is_head_param(n)
    ]
    backbone_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad and not _is_head_param(n)
    ]
    param_groups = [
        {"params": backbone_params, "lr": lr},
        {"params": head_params, "lr": lr * head_lr_mult},
    ]

    if optimizer_name == "SGD":
        optimizer = torch.optim.SGD(
            param_groups, momentum=momentum, weight_decay=weight_decay
        )
    elif optimizer_name == "Adam":
        optimizer = torch.optim.Adam(param_groups, weight_decay=weight_decay)
    elif optimizer_name == "AdamW":
        optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    else:
        raise ValueError(f"지원하지 않는 optimizer: {optimizer_name}")

    scheduler_name = train_cfg.get("scheduler", "cosine")
    epochs = train_cfg.get("epochs", 50)
    warmup_epochs = train_cfg.get("warmup_epochs", 0)

    if scheduler_name == "cosine":
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(epochs - warmup_epochs, 1)
        )
    elif scheduler_name == "step":
        main_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=train_cfg.get("step_size", 10), gamma=0.1
        )
    else:
        main_scheduler = None

    if warmup_epochs > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, main_scheduler] if main_scheduler else [warmup],
            milestones=[warmup_epochs],
        )
    else:
        scheduler = main_scheduler

    output_dir = Path(project_dir) / exp_name
    weights_dir = output_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, output_dir / "config.yaml")

    freeze_epochs = train_cfg.get("freeze_backbone_epochs", 0)
    if freeze_epochs > 0:
        for name, p in model.named_parameters():
            if not _is_head_param(name):
                p.requires_grad_(False)
        print(f"Backbone frozen for first {freeze_epochs} epochs (detection head only)")

    best_map = 0.0
    history = []

    for epoch in range(1, epochs + 1):
        if freeze_epochs > 0 and epoch == freeze_epochs + 1:
            for p in model.parameters():
                p.requires_grad_(True)
            # RPN 폭발 방지: unfreeze 시점에 backbone LR을 매우 낮게 고정
            unfreeze_lr = train_cfg.get("unfreeze_backbone_lr", 1e-4)
            optimizer.param_groups[0]["lr"] = unfreeze_lr
            print(
                f"Epoch {epoch}: Backbone unfrozen "
                f"(backbone lr: {unfreeze_lr:.1e}, head lr: {optimizer.param_groups[1]['lr']:.2e})"
            )

        t0 = time.time()
        avg_loss, loss_components = train_one_epoch(model, optimizer, train_loader, device)
        elapsed = time.time() - t0

        if device.type == "mps":
            torch.mps.empty_cache()

        if scheduler:
            scheduler.step()

        metrics = evaluate_coco(model, val_loader, device)
        if device.type == "mps":
            torch.mps.empty_cache()
        metrics["epoch"] = epoch
        metrics["train_loss"] = avg_loss
        metrics["lr"] = optimizer.param_groups[0]["lr"]
        metrics["lr_head"] = optimizer.param_groups[1]["lr"]
        metrics["time"] = round(elapsed, 1)
        history.append(metrics)

        map75_95 = metrics["map"]
        components_str = " | ".join(f"{k}: {v:.3f}" for k, v in loss_components.items())
        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"loss: {avg_loss:.4f} ({components_str}) | "
            f"mAP@75: {metrics['map75']:.4f} | "
            f"mAP@75:95: {map75_95:.4f} | "
            f"lr: {metrics['lr']:.2e} / head: {metrics['lr_head']:.2e} | "
            f"{elapsed:.0f}s"
        )

        if map75_95 > best_map:
            best_map = map75_95
            torch.save(model.state_dict(), weights_dir / "best.pt")

    torch.save(model.state_dict(), weights_dir / "last.pt")

    final_metrics = {
        "best_map75_95": best_map,
        "final_map75": history[-1]["map75"],
        "final_map75_95": history[-1]["map"],
        "epochs": epochs,
        "model": config["model"]["name"],
        "history": history,
    }

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2)
    print(f"\n결과 저장: {output_dir}")

    return final_metrics
