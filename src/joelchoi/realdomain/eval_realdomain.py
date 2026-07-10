"""실도메인 검증셋 평가 하네스.

라벨된 실촬영 셋에서 두 가지를 측정한다:
  1) 클래스-agnostic 검출: 정체성 무시, 알약을 잘 찾나 (mAP@0.5, mAP@0.5:0.95, recall)
  2) 식별 정확도: IoU≥0.5로 매칭된 박스 중 클래스가 맞은 비율

GT 라벨은 YOLO txt(정규화 xywh, 클래스=모델 인덱스, autolabel 초안을 사람이 교정).

사용:
    python -m src.joelchoi.realdomain.eval_realdomain \
        --weights experiments/joelchoi/exp011_yolo11n_aug/weights/best.pt \
        --images data/realdomain_eval/images \
        --labels data/realdomain_eval/labels
"""

import argparse
from pathlib import Path


def _load_gt(label_path: Path, W: int, H: int):
    """YOLO txt → (boxes_xyxy 픽셀 list, cls list)."""
    boxes, cls = [], []
    if not label_path.exists():
        return boxes, cls
    for line in label_path.read_text().splitlines():
        p = line.split()
        if len(p) != 5:
            continue
        c, cx, cy, w, h = int(p[0]), *map(float, p[1:])
        x1 = (cx - w / 2) * W
        y1 = (cy - h / 2) * H
        x2 = (cx + w / 2) * W
        y2 = (cy + h / 2) * H
        boxes.append([x1, y1, x2, y2])
        cls.append(c)
    return boxes, cls


def _iou(a, b):
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def evaluate(
    weights: str,
    images: str,
    labels: str,
    conf: float = 0.25,
    iou: float = 0.45,
    imgsz: int = 640,
    match_iou: float = 0.5,
) -> dict:
    import torch
    from PIL import Image
    from ultralytics import YOLO
    from torchmetrics.detection import MeanAveragePrecision

    images = Path(images)
    labels = Path(labels)
    exts = ("*.jpg", "*.jpeg", "*.png")
    paths = sorted(p for e in exts for p in images.glob(e))
    print(f"평가 이미지: {len(paths)}장")

    model = YOLO(str(weights))
    metric = MeanAveragePrecision(iou_type="bbox")  # 기본 0.5:0.95 + map_50

    n_gt = n_matched = n_id_correct = 0
    gt_has_id = False  # GT 라벨에 실제 정체성(class>0)이 있는지

    results = model.predict(
        source=[str(p) for p in paths],
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        stream=True,
        verbose=False,
    )
    for p, res in zip(paths, results):
        with Image.open(p) as im:
            W, H = im.size
        gt_boxes, gt_cls = _load_gt(labels / f"{p.stem}.txt", W, H)

        b = res.boxes
        if b is None or len(b) == 0:
            pred_boxes, pred_cls, pred_scores = [], [], []
        else:
            pred_boxes = b.xyxy.tolist()
            pred_cls = [int(c) for c in b.cls.tolist()]
            pred_scores = b.conf.tolist()

        # --- 검출 mAP (클래스-agnostic: 라벨 전부 0) ---
        metric.update(
            [
                {
                    "boxes": torch.tensor(pred_boxes).reshape(-1, 4).float(),
                    "scores": torch.tensor(pred_scores).float(),
                    "labels": torch.zeros(len(pred_boxes), dtype=torch.long),
                }
            ],
            [
                {
                    "boxes": torch.tensor(gt_boxes).reshape(-1, 4).float(),
                    "labels": torch.zeros(len(gt_boxes), dtype=torch.long),
                }
            ],
        )

        # --- 식별 정확도 (greedy IoU 매칭) ---
        n_gt += len(gt_boxes)
        gt_has_id = gt_has_id or any(c > 0 for c in gt_cls)
        used = set()
        order = sorted(range(len(pred_boxes)), key=lambda i: -pred_scores[i])
        for gi, gb in enumerate(gt_boxes):
            best, bj = match_iou, -1
            for pj in order:
                if pj in used:
                    continue
                v = _iou(gb, pred_boxes[pj])
                if v >= best:
                    best, bj = v, pj
            if bj >= 0:
                used.add(bj)
                n_matched += 1
                if pred_cls[bj] == gt_cls[gi]:
                    n_id_correct += 1

    m = metric.compute()
    det_map = float(m["map"])
    det_map50 = float(m["map_50"])
    recall = n_matched / max(n_gt, 1)
    id_acc = (n_id_correct / max(n_matched, 1)) if gt_has_id else None

    print("\n── 실도메인 성능 ─────────────────────")
    print(f"검출(클래스-agnostic)  mAP@0.5     : {det_map50:.4f}")
    print(f"검출(클래스-agnostic)  mAP@0.5:0.95: {det_map:.4f}")
    print(
        f"로컬라이제이션 recall(IoU≥{match_iou}) : {recall:.4f}  ({n_matched}/{n_gt})"
    )
    if gt_has_id:
        print(
            f"식별 정확도(매칭 박스 중)          : {id_acc:.4f}  ({n_id_correct}/{n_matched})"
        )
    else:
        print(
            "식별 정확도                        : N/A (GT가 class-agnostic, 정체성 라벨 없음)"
        )
    print(
        "\n해석: recall이 낮으면 '못 찾는' 문제(합성·증강 필요), "
        "recall은 높은데 식별이 낮으면 '못 맞추는' 문제(임베딩 식별 필요)."
    )
    return {
        "map50": det_map50,
        "map": det_map,
        "recall": recall,
        "id_acc": id_acc,
        "n_gt": n_gt,
        "n_matched": n_matched,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--match-iou", type=float, default=0.5)
    a = ap.parse_args()
    evaluate(a.weights, a.images, a.labels, a.conf, a.iou, a.imgsz, a.match_iou)
