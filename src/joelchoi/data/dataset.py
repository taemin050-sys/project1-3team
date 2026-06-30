"""torchvision 모델용 COCO 형식 Dataset.

Augmentation은 Albumentations를 사용하여 bbox와 함께 변환.
"""

from collections import defaultdict

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

import albumentations as A


class PillDetectionDataset(Dataset):
    def __init__(self, coco: dict, augment: A.Compose | None = None):
        self.images = coco["images"]
        self.categories = coco["categories"]
        self.augment = augment
        self.to_tensor = ToTensor()

        self.anns_by_img = defaultdict(list)
        for ann in coco["annotations"]:
            self.anns_by_img[ann["image_id"]].append(ann)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        image = Image.open(img_info["file_name"]).convert("RGB")

        anns = self.anns_by_img.get(img_info["id"], [])

        boxes = []
        labels = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(ann["category_id"])

        if self.augment is not None:
            out = self.augment(image=np.array(image), bboxes=boxes, labels=labels)
            image = Image.fromarray(out["image"])
            boxes = out["bboxes"]
            labels = out["labels"]

        image = self.to_tensor(image)

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        return image, {"boxes": boxes, "labels": labels}


def _bbox_params():
    return A.BboxParams(
        coord_format="pascal_voc",  # torchvision 모델들이 해당형식을 요구함
        label_fields=["labels"],
        min_visibility=0.3,
        clip_after_transform=True,
    )


def get_train_augment(img_size: int | None = None):
    transforms = []
    if img_size:
        transforms.append(A.LongestMaxSize(max_size=img_size))
    transforms.extend([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.Affine(scale=(0.9, 1.1), translate_percent=(0, 0.1), p=0.5),
    ])
    return A.Compose(transforms, bbox_params=_bbox_params())


def get_val_augment(img_size: int | None = None):
    if img_size is None:
        return None
    return A.Compose(
        [A.LongestMaxSize(max_size=img_size)],
        bbox_params=_bbox_params(),
    )


def collate_fn(batch):
    return tuple(zip(*batch))
