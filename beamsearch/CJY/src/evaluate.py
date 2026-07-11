"""공통 평가 모듈 — torchvision 모델용 mAP 계산."""

import numpy as np
import torch
from torchmetrics.detection import MeanAveragePrecision


@torch.no_grad()
def evaluate_coco(model, data_loader, device) -> dict:
    """torchvision detection 모델의 mAP를 계산.

    IoU threshold: 0.75~0.95 (step 0.05)

    Returns:
        dict with map75, map (mean over IoU 0.75:0.95)
    """
    model.eval()
    custom_iou = np.arange(0.75, 0.96, 0.05).tolist()
    metric = MeanAveragePrecision(iou_type="bbox", iou_thresholds=custom_iou)

    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        outputs = model(images)

        preds = []
        gts = []

        for output, target in zip(outputs, targets):
            preds.append({
                "boxes": output["boxes"].cpu(),
                "scores": output["scores"].cpu(),
                "labels": output["labels"].cpu(),
            })
            gts.append({
                "boxes": target["boxes"].cpu()
                if isinstance(target["boxes"], torch.Tensor)
                else torch.as_tensor(target["boxes"]),
                "labels": target["labels"].cpu()
                if isinstance(target["labels"], torch.Tensor)
                else torch.as_tensor(target["labels"]),
            })

        metric.update(preds, gts)

    result = metric.compute()

    return {
        # "map50": float(result["map_50"]),
        "map75": float(result["map_75"]),
        "map": float(result["map"]),
    }
