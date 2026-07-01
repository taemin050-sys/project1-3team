"""torchvision 모델용 COCO 형식 Dataset.

Augmentation은 Albumentations를 사용하여 bbox와 함께 변환.

핵심(라벨 리매핑):
COCO category_id는 K-코드(예: 1900, 24850)로 연속적이지 않다. torchvision
detection 모델은 라벨이 [1 .. num_classes-1] 범위여야 하고 0은 배경이다.
따라서 category_id를 **연속 라벨(1..N)** 로 리매핑한다. 리매핑을 안 하면
라벨 값이 클래스 수를 초과해 CUDA assert/NaN이 난다.
"""

from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

import albumentations as A


class PillDetectionDataset(Dataset):
    def __init__(
        self,
        coco: dict,
        augment: A.Compose | None = None,
        cat2label: dict[int, int] | None = None,
    ):
        self.images = coco["images"]
        self.categories = coco["categories"]
        self.augment = augment
        self.to_tensor = ToTensor()

        # category_id → 연속 라벨(1..N). 0은 배경 예약.
        # train/val이 동일 매핑을 쓰도록 외부 주입 가능.
        if cat2label is None:
            cat_ids = sorted(c["id"] for c in coco["categories"])
            cat2label = {cid: i + 1 for i, cid in enumerate(cat_ids)}
        self.cat2label = cat2label
        self.label2cat = {v: k for k, v in cat2label.items()}

        self.anns_by_img = defaultdict(list)
        for ann in coco["annotations"]:
            self.anns_by_img[ann["image_id"]].append(ann)

    @property
    def num_classes(self) -> int:
        """배경 포함 클래스 수 (N+1)."""
        return len(self.cat2label) + 1

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        image = Image.open(img_info["file_name"]).convert("RGB")
        w0, h0 = image.size

        anns = self.anns_by_img.get(img_info["id"], [])

        boxes = []
        labels = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            cid = ann["category_id"]
            if cid not in self.cat2label:
                continue  # 매핑에 없는 클래스는 무시
            boxes.append([x, y, x + w, y + h])
            labels.append(self.cat2label[cid])

        if self.augment is not None:
            out = self.augment(image=np.array(image), bboxes=boxes, labels=labels)
            image = Image.fromarray(out["image"])
            boxes = list(out["bboxes"])
            labels = list(out["labels"])
            w0, h0 = image.size

        # 좌표를 이미지 경계로 클립하고 degenerate box(폭/높이<1px) 제거.
        # (증강 후 남는 무효 박스는 Faster R-CNN 등에서 손실을 NaN으로 만든다.)
        clean_boxes, clean_labels = [], []
        for (x1, y1, x2, y2), lb in zip(boxes, labels):
            x1 = max(0.0, min(float(x1), w0))
            y1 = max(0.0, min(float(y1), h0))
            x2 = max(0.0, min(float(x2), w0))
            y2 = max(0.0, min(float(y2), h0))
            if x2 - x1 >= 1.0 and y2 - y1 >= 1.0:
                clean_boxes.append([x1, y1, x2, y2])
                clean_labels.append(int(lb))

        image = self.to_tensor(image)

        if len(clean_boxes) == 0:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes_t = torch.tensor(clean_boxes, dtype=torch.float32)
            labels_t = torch.tensor(clean_labels, dtype=torch.int64)

        return image, {
            "boxes": boxes_t,
            "labels": labels_t,
            "image_id": torch.tensor([img_info["id"]]),
        }


def build_cat2label(coco: dict) -> dict[int, int]:
    """train/val 공유용 category_id→라벨(1..N) 매핑 생성."""
    cat_ids = sorted(c["id"] for c in coco["categories"])
    return {cid: i + 1 for i, cid in enumerate(cat_ids)}


def _bbox_params():
    # pascal_voc = [x_min, y_min, x_max, y_max].
    # Albumentations 버전에 따라 인자명이 'format' 또는 'coord_format'이라 방어적으로 처리.
    common = dict(label_fields=["labels"], min_visibility=0.3)
    for key in ("format", "coord_format"):
        try:
            return A.BboxParams(**{key: "pascal_voc"}, **common)
        except TypeError:
            continue
    # 마지막 수단: 위치 인자
    return A.BboxParams("pascal_voc", **common)


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
