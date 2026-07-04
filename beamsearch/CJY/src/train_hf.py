"""HuggingFace transformers 기반 DETR 계열(RF-DETR) 학습 루프.

torchvision 계열(`src/train.py`)과 호출 인터페이스를 맞춰서 같은 노트북 루프에서
그대로 섞어 쓸 수 있게 만든다:

    train_rfdetr(config, train_coco, val_coco, project_dir) -> metrics dict

RF-DETR(`transformers.RfDetrForObjectDetection`)는 torchvision 모델과 forward
시그니처가 다르다(``pixel_values``/``pixel_mask``/``labels``, DETR 스타일 COCO
어노테이션 입력). Multi-Scale Deformable Attention을 쓰지만, 이 transformers
버전에서는 ``config.disable_custom_kernels`` 기본값이 True라 CUDA 커널 컴파일
없이 순수 PyTorch로 동작한다 — Mac(MPS)/CPU에서도 실행 가능하다(다만 컴파일된
CUDA 커널 대비 느릴 수 있다). DINO-DETR/Sparse R-CNN은 각각 커스텀 CUDA 커널
필수, HuggingFace 미지원이라 이 저장소의 Mac 환경에는 맞지 않아 대신 RF-DETR을
transformer 계열 대표로 추가했다.

주의: 이 모듈은 실제 알약 이미지/GPU 환경에서 end-to-end로 실행 검증되지 않았다
(설치된 transformers 소스 코드 기준으로 API를 맞췄을 뿐). 처음 돌릴 때는 config의
``training.epochs``를 1~2로 낮추고 이미지 수도 작은 subset으로 먼저 스모크 테스트할 것.
"""

import json
import os
import time
from collections import defaultdict
from pathlib import Path

# RF-DETR은 Multi-Scale Deformable Attention의 샘플링에 grid_sample을 쓰는데,
# 이 연산의 backward(``aten::grid_sampler_2d_backward``)가 PyTorch MPS 백엔드에
# 아직 구현돼 있지 않다(forward는 되는데 backward만 없음 — 2026-07 기준 PyTorch
# 공식 이슈로 남아있는 gap). PYTORCH_ENABLE_MPS_FALLBACK=1을 켜두면 이 연산만
# 자동으로 CPU로 폴백되고 나머지는 그대로 MPS에서 돈다(그 연산만 느려짐, 전체를
# CPU로 내리는 것보다 훨씬 낫다). torch가 이 파일에서 처음 import되기 전에
# 설정해야 확실히 적용되므로 최상단에 둔다. 이미 다른 셀/모듈에서 torch를 먼저
# import한 뒤라도 실제로는 대개 적용되지만(연산이 dispatch되는 시점에 읽음),
# 가장 안전하게 하려면 노트북 첫 셀에서도 동일하게 설정해 둘 것.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchmetrics.detection import MeanAveragePrecision
from tqdm.auto import tqdm

from src.utils import get_device, load_hf_token, save_config, set_seed

DEFAULT_CHECKPOINT = "Roboflow/rf-detr-base"


class _CocoDetectionDataset(Dataset):
    """COCO dict -> (PIL.Image, DETR 스타일 target) 페어를 반환하는 Dataset.

    target 형식은 HuggingFace 이미지 프로세서(`RfDetrImageProcessor` 등)가
    바로 소비할 수 있는 COCO 어노테이션 딕셔너리 리스트다.
    """

    def __init__(self, coco: dict, cat2label: dict[int, int]):
        self.images = coco["images"]
        self.cat2label = cat2label
        self.anns_by_img: dict[int, list[dict]] = defaultdict(list)
        for a in coco["annotations"]:
            self.anns_by_img[a["image_id"]].append(a)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img_info = self.images[idx]
        image = Image.open(img_info["file_name"]).convert("RGB")

        anns = []
        for a in self.anns_by_img.get(img_info["id"], []):
            if a["category_id"] not in self.cat2label:
                continue  # train/val 공유 매핑에 없는 클래스는 무시(다른 모델들과 동일 관례)
            bbox = a["bbox"]
            anns.append({
                "image_id": img_info["id"],
                "category_id": self.cat2label[a["category_id"]],
                "bbox": bbox,
                "area": a.get("area", bbox[2] * bbox[3]),
                "iscrowd": a.get("iscrowd", 0),
            })

        target = {"image_id": img_info["id"], "annotations": anns}
        return image, target


class _RfDetrCollator:
    """DataLoader collate_fn을 모듈 레벨의 picklable 클래스로 분리.

    이전에는 `train_rfdetr()` 안에서 `def collate_fn(...)`으로 클로저를 만들어
    썼는데, ``num_workers > 0``이면 DataLoader가 워커 프로세스로 collate_fn을
    피클링해서 넘겨야 한다 — 로컬(중첩) 함수는 pickle이 안 돼서
    ``PicklingError: Can't pickle local object ...collate_fn``으로 즉시 죽는다
    (Mac/Windows처럼 spawn 방식을 쓰는 환경에서 특히 잘 터진다). 클래스 인스턴스는
    피클 가능하므로 이 형태로 바꿨다(`processor`도 피클 가능한 일반 객체).
    """

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        images, targets = zip(*batch)
        encoded = self.processor(
            images=list(images), annotations=list(targets), return_tensors="pt"
        )
        # 평가용 원본 이미지 크기(후처리 시 박스를 원본 픽셀 좌표로 되돌리는 데 필요)
        orig_sizes = [img.size[::-1] for img in images]  # (H, W)
        image_ids = [t["image_id"] for t in targets]
        return encoded, orig_sizes, image_ids


def _rf_detr_compatible_size(checkpoint: str, img_size: int) -> int:
    """RF-DETR(DINOv2 backbone)의 windowed attention과 호환되는 정사각 입력 크기로 반올림.

    백본이 이미지를 ``patch_size``짜리 패치로 나눈 뒤 ``num_windows x num_windows``
    윈도우로 다시 묶어 attention을 계산하므로(예: patch_size=14, num_windows=4),
    한 변의 패치 개수가 ``num_windows``로 나누어떨어져야 한다 — 즉 이미지 한 변의
    길이가 ``patch_size * num_windows``의 배수여야 한다. 체크포인트마다 값이 다를
    수 있어 하드코딩하지 않고 실제 config에서 읽는다.
    """
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(checkpoint)
    backbone_cfg = getattr(cfg, "backbone_config", None)
    patch_size = (
        getattr(backbone_cfg, "patch_size", 14) if backbone_cfg is not None else 14
    )
    num_windows = (
        getattr(backbone_cfg, "num_windows", 4) if backbone_cfg is not None else 4
    )
    if isinstance(patch_size, (list, tuple)):
        patch_size = patch_size[0]

    size_multiple = max(1, int(patch_size) * int(num_windows))
    if img_size % size_multiple == 0:
        return img_size
    rounded = round(img_size / size_multiple) * size_multiple
    return max(size_multiple, rounded)


def build_cat2label(coco: dict) -> dict[int, int]:
    """category_id → 0..N-1 라벨(DETR 계열은 배경 클래스를 별도로 두지 않는다).

    torchvision 쪽(`src/data/dataset.py::build_cat2label`)은 1..N(0=배경)을 쓰지만
    HuggingFace DETR 계열은 0..N-1을 쓰는 관례라 별도로 둔다. train/val이 반드시
    같은 매핑을 공유해야 하므로(다른 모델들과 동일한 이유) train 기준으로 만든
    매핑을 val에도 그대로 넘겨써야 한다.
    """
    cat_ids = sorted({int(c["id"]) for c in coco["categories"]})
    return {cid: i for i, cid in enumerate(cat_ids)}


def train_rfdetr(
    config: dict,
    train_coco: dict,
    val_coco: dict,
    project_dir: str | Path,
    checkpoint: str = DEFAULT_CHECKPOINT,
) -> dict:
    """config 기반으로 RF-DETR(HuggingFace) 모델 학습.

    Args:
        config: 실험 config dict. `src/train.py::train_torchvision`과 동일한
            섹션(experiment/data/model/training)을 쓴다. ``training.accumulate_grad_batches``
            도 동일한 의미로 지원한다(메모리 절약용 그래디언트 누적).
        train_coco, val_coco: COCO dict. train/val이 동일한 category 매핑을 쓰도록
            train 기준 매핑을 val에도 적용한다(다른 학습 함수들과 동일 원칙).
        project_dir: 결과 저장 경로(다른 실험들과 같은 `experiments/` 트리 아래).
        checkpoint: 사전학습 체크포인트. 기본 "Roboflow/rf-detr-base".

    Returns:
        다른 학습 함수들과 동일한 스키마의 metrics dict
        (best_map75_95, final_map75, final_map75_95, epochs, model, history).
    """
    # transformers는 무거운 의존성이라 실제로 이 함수를 호출할 때만 임포트한다
    # (torchvision-only 노트북/환경에서 불필요한 임포트 실패를 피하기 위함).
    from transformers import AutoImageProcessor, RfDetrForObjectDetection

    # .env의 HG_TOKEN을 huggingface_hub가 인식하는 HF_TOKEN으로 등록(있으면).
    # 공개 체크포인트는 토큰 없이도 동작하지만, rate limit 회피/향후 gated
    # 체크포인트 대비용으로 있으면 항상 쓴다.
    load_hf_token()

    exp_name = config["experiment"]["name"]
    seed = config["experiment"].get("seed", 42)
    set_seed(seed)

    device = get_device()
    print(f"Device: {device}")

    train_cfg = config["training"]
    img_size = config.get("data", {}).get("img_size", 640)
    accumulate_grad_batches = max(1, int(train_cfg.get("accumulate_grad_batches", 1)))

    cat2label = build_cat2label(train_coco)
    label2cat = {v: k for k, v in cat2label.items()}
    num_labels = len(cat2label)
    id2label = {i: str(label2cat[i]) for i in range(num_labels)}
    label2id = {v: k for k, v in id2label.items()}

    rf_img_size = _rf_detr_compatible_size(checkpoint, img_size)
    if rf_img_size != img_size:
        print(
            f"RF-DETR: img_size {img_size} → {rf_img_size}로 반올림 "
            f"(DINOv2 windowed attention이 patch_size*num_windows의 배수인 정사각 입력을 요구함)"
        )

    # size를 {"height", "width"}로(=shortest/longest_edge 대신) 지정해 종횡비를 유지하지
    # 않는 강제 정사각 리사이즈로 만든다. 기존에 shortest_edge/longest_edge로 지정하면
    # 원본 사진마다 리사이즈 후 크기가 제각각이라 배치 패딩(do_pad) 시 최종 (H, W)가
    # 배치마다 달라지고, 그 크기가 patch_size*num_windows(기본 14*4=56)의 배수가 아니면
    # DINOv2 backbone의 windowed-attention reshape에서
    # "shape '[...]' is invalid for input of size ..." RuntimeError로 죽는다.
    # 모든 이미지를 동일한 정사각 크기로 강제 리사이즈하면 배치 내 크기가 항상
    # 일정해서 패딩이 사실상 no-op이 되고, 그 크기를 56의 배수로 맞춰 놓으면
    # windowed attention의 reshape도 항상 성립한다.
    processor = AutoImageProcessor.from_pretrained(
        checkpoint, size={"height": rf_img_size, "width": rf_img_size}
    )

    model = RfDetrForObjectDetection.from_pretrained(
        checkpoint,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,  # 사전학습 head(클래스 수 다름) 대신 새 head로 초기화
        disable_custom_kernels=True,  # CUDA 커널 컴파일 없이 순수 PyTorch 경로 사용(Mac/CPU 호환)
    )
    model = model.to(device)

    train_ds = _CocoDetectionDataset(train_coco, cat2label)
    val_ds = _CocoDetectionDataset(val_coco, cat2label)
    collate_fn = _RfDetrCollator(processor)

    batch_size = train_cfg.get("batch_size", 4)
    # 기본값 0(메인 프로세스에서 로딩): HF 이미지 프로세서 + Mac(spawn) 조합에서
    # num_workers>0가 잘 터지는 사례가 많아 안전한 값을 기본으로 한다.
    # collate_fn을 picklable하게 고쳤으니 num_workers>0도 이제 동작은 하지만,
    # 처음 켜볼 땐 작은 값으로 검증할 것.
    num_workers = train_cfg.get("num_workers", 0)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    # 평가용 GT 룩업: image_id -> {"boxes": xyxy 절대좌표, "labels": Tensor}.
    # HF 프로세서가 내부적으로 만드는 정규화 박스(리사이즈/패딩 프레임 기준)를 역변환해서
    # 쓰지 않고, 원본 COCO 어노테이션에서 직접 절대좌표를 만든다 — 다른 모델들의
    # `evaluate_coco`와 동일한 방식이라 프레임/패딩 가정 차이로 인한 오차가 없다.
    val_gt_lookup = _build_gt_lookup(val_coco, cat2label)

    lr = train_cfg.get("lr", 1e-4)
    weight_decay = train_cfg.get("weight_decay", 1e-4)
    optimizer_name = train_cfg.get("optimizer", "AdamW")
    if optimizer_name == "AdamW":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
    elif optimizer_name == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay
        )

    epochs = train_cfg.get("epochs", 30)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1)
    )

    output_dir = Path(project_dir) / exp_name
    weights_dir = output_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, output_dir / "config.yaml")

    # ── 재개(resume) 지원 ──────────────────────────────────────────────────
    # config의 training.resume: true면, 이전에 끊긴 학습이 매 epoch마다 저장한
    # checkpoint.pt(모델+옵티마이저+스케줄러 상태 전부)를 불러와 이어서 학습한다.
    # resume: false/미지정이면 기존과 동일하게 항상 처음부터 새로 시작한다
    # (기존 config 파일들은 이 키가 없으므로 동작이 바뀌지 않는다).
    resume = bool(train_cfg.get("resume", False))
    checkpoint_path = weights_dir / "checkpoint.pt"
    best_map = 0.0
    history = []
    start_epoch = 1

    if resume and checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        best_map = ckpt["best_map"]
        history = ckpt["history"]
        start_epoch = ckpt["epoch"] + 1
        print(
            f"체크포인트에서 재개: {checkpoint_path} → epoch {start_epoch}부터 "
            f"(지금까지 best mAP@75:95={best_map:.4f})"
        )
        if start_epoch > epochs:
            print(
                f"이미 목표 epoch({epochs})까지 끝나 있어 재학습 없이 저장된 결과를 그대로 씁니다."
            )
    elif resume:
        print(
            f"resume=True지만 체크포인트가 없어({checkpoint_path}) 처음부터 시작합니다."
        )

    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        num_batches = 0
        optimizer.zero_grad()

        for step, (encoded, _orig_sizes, _image_ids) in enumerate(
            tqdm(train_loader), start=1
        ):
            encoded = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in encoded.items()
            }
            if "labels" in encoded:
                encoded["labels"] = [
                    {
                        k2: (v2.to(device) if isinstance(v2, torch.Tensor) else v2)
                        for k2, v2 in lb.items()
                    }
                    for lb in encoded["labels"]
                ]

            outputs = model(**encoded)
            loss = outputs.loss

            if loss is None or not torch.isfinite(loss):
                optimizer.zero_grad()
                continue

            (loss / accumulate_grad_batches).backward()

            if step % accumulate_grad_batches == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item()
            num_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(num_batches, 1)
        elapsed = time.time() - t0

        metrics = _evaluate_rfdetr(model, processor, val_loader, device, val_gt_lookup)
        metrics["epoch"] = epoch
        metrics["train_loss"] = avg_loss
        metrics["lr"] = optimizer.param_groups[0]["lr"]
        metrics["time"] = round(elapsed, 1)
        history.append(metrics)

        map75_95 = metrics["map"]
        print(
            f"Epoch {epoch:3d}/{epochs} | loss: {avg_loss:.4f} | "
            f"mAP@75: {metrics['map75']:.4f} | mAP@75:95: {map75_95:.4f} | "
            f"lr: {metrics['lr']:.2e} | {elapsed:.0f}s"
        )

        if map75_95 > best_map:
            best_map = map75_95
            torch.save(model.state_dict(), weights_dir / "best.pt")

        # 매 epoch마다 재개용 전체 상태 저장(모델+옵티마이저+스케줄러+history).
        # best.pt/last.pt는 순수 가중치만(추론용, 기존 관례 유지) — checkpoint.pt는
        # 이 함수 재개 전용이라 형식이 다르다.
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_map": best_map,
                "history": history,
            },
            checkpoint_path,
        )

    torch.save(model.state_dict(), weights_dir / "last.pt")

    final_metrics = {
        "best_map75_95": best_map,
        "final_map75": history[-1]["map75"] if history else 0.0,
        "final_map75_95": history[-1]["map"] if history else 0.0,
        "epochs": epochs,
        "model": config["model"]["name"],
        "history": history,
    }

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2)
    print(f"\n결과 저장: {output_dir}")

    return final_metrics


def _build_gt_lookup(coco: dict, cat2label: dict[int, int]) -> dict[int, dict]:
    """image_id -> {"boxes": xyxy 절대좌표 Tensor[N,4], "labels": Tensor[N]}.

    원본 COCO 어노테이션(절대 픽셀, xywh)에서 직접 만든다 — HF 프로세서가 내부
    학습용으로 만드는 정규화 박스(리사이즈/패딩 프레임 기준)를 evaluation에서
    역변환해 쓰지 않기 위함(패딩 여부에 따라 오차가 생길 수 있어 원본을 그대로 씀).
    다른 모델들의 `src/evaluate.py::evaluate_coco`와 동일한 GT 표현 방식이다.
    """
    anns_by_img: dict[int, list[dict]] = defaultdict(list)
    for a in coco["annotations"]:
        anns_by_img[a["image_id"]].append(a)

    lookup = {}
    for img in coco["images"]:
        anns = anns_by_img.get(img["id"], [])
        boxes, labels = [], []
        for a in anns:
            if a["category_id"] not in cat2label:
                continue
            x, y, w, h = a["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(cat2label[a["category_id"]])
        lookup[img["id"]] = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
        }
    return lookup


@torch.no_grad()
def _evaluate_rfdetr(
    model, processor, data_loader, device, gt_lookup: dict[int, dict]
) -> dict:
    """RF-DETR mAP 계산. IoU 0.75~0.95(step 0.05) — 다른 모델들과 동일 기준."""
    model.eval()
    custom_iou = np.arange(0.75, 0.96, 0.05).tolist()
    metric = MeanAveragePrecision(iou_type="bbox", iou_thresholds=custom_iou)

    for encoded, orig_sizes, image_ids in data_loader:
        inputs = {
            k: v.to(device)
            for k, v in encoded.items()
            if k in ("pixel_values", "pixel_mask")
        }
        outputs = model(**inputs)

        target_sizes = torch.tensor(orig_sizes, device=device)
        results = processor.post_process_object_detection(
            outputs, threshold=0.0, target_sizes=target_sizes
        )

        preds, gts = [], []
        for res, img_id in zip(results, image_ids):
            preds.append({
                "boxes": res["boxes"].cpu(),
                "scores": res["scores"].cpu(),
                "labels": res["labels"].cpu(),
            })
            gt = gt_lookup.get(
                img_id,
                {
                    "boxes": torch.zeros((0, 4)),
                    "labels": torch.zeros((0,), dtype=torch.int64),
                },
            )
            gts.append(gt)

        metric.update(preds, gts)

    result = metric.compute()
    return {
        "map75": float(result["map_75"]),
        "map": float(result["map"]),
    }
