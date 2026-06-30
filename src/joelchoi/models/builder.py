"""Config 기반 모델 팩토리."""

import torch.nn as nn
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2,
    fcos_resnet50_fpn,
    retinanet_resnet50_fpn_v2,
    ssd300_vgg16,
    ssdlite320_mobilenet_v3_large,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.ssd import SSDClassificationHead
from torchvision.models.detection._utils import retrieve_out_channels


def _replace_ssd_head(model: nn.Module, num_classes: int, input_size: int = 300):
    in_channels = retrieve_out_channels(model.backbone, (input_size, input_size))
    num_anchors = model.anchor_generator.num_anchors_per_location()
    model.head.classification_head = SSDClassificationHead(
        in_channels, num_anchors, num_classes
    )


def build_model(config: dict, num_classes: int) -> nn.Module:
    """config의 model 섹션으로 torchvision detection 모델 생성.

    Args:
        config: 전체 실험 config (model 키 포함)
        num_classes: 클래스 수 (배경 포함)

    Returns:
        torchvision detection 모델
    """
    model_cfg = config["model"]
    name = model_cfg["name"]
    pretrained = model_cfg.get("pretrained", True)

    if name == "fasterrcnn_resnet50":
        weights = "DEFAULT" if pretrained else None
        img_size = config.get("data", {}).get("img_size")
        if img_size:
            model = fasterrcnn_resnet50_fpn_v2(
                weights=weights, min_size=img_size, max_size=img_size
            )
        else:
            model = fasterrcnn_resnet50_fpn_v2(weights=weights)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    elif name == "ssd300":
        weights = "DEFAULT" if pretrained else None
        model = ssd300_vgg16(weights=weights)
        _replace_ssd_head(model, num_classes)

    elif name == "ssdlite320":
        weights = "DEFAULT" if pretrained else None
        model = ssdlite320_mobilenet_v3_large(weights=weights)
        _replace_ssd_head(model, num_classes)

    elif name == "retinanet_resnet50":
        weights = "DEFAULT" if pretrained else None
        model = retinanet_resnet50_fpn_v2(weights=weights, num_classes=num_classes)

    elif name == "fcos_resnet50":
        weights = "DEFAULT" if pretrained else None
        model = fcos_resnet50_fpn(weights=weights, num_classes=num_classes)

    else:
        raise ValueError(
            f"지원하지 않는 모델: {name}. "
            f"사용 가능: fasterrcnn_resnet50, ssd300, ssdlite320, "
            f"retinanet_resnet50, fcos_resnet50"
        )

    return model
