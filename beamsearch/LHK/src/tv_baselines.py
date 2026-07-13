"""torchvision 검출 베이스라인 (LHK/MPS): Faster R-CNN · RetinaNet · FCOS.
동일 fold0 · 동일 mAP@[0.75:0.95] 하니스. 예산: min_size=512, 20ep (문서화)."""

import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    retinanet_resnet50_fpn,
    fcos_resnet50_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from torchvision.models.detection.fcos import FCOSClassificationHead
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

DEV = "mps"
W, H = 976, 1280
NC = 56
BASE = Path("/Users/macbook/dev/learning/codeit/01_Proj_HealthEat_cla")
LHK = BASE / "project1-3team/beamsearch/LHK"
SSOT, YD, RUNS = LHK / "data/processed", LHK / "data/yolo", LHK / "runs"
m2c = {
    int(k): v
    for k, v in json.load(open(SSOT / "class_map.json"))[
        "model_index_to_category_id"
    ].items()
}
cocoGt = COCO(str(RUNS / "val_gt.json"))
val_imgs = sorted((YD / "images/val").glob("*.png"))
nid = {p.name: i + 1 for i, p in enumerate(val_imgs)}


class DS(Dataset):
    def __init__(self, split, off):
        self.imgs = sorted((YD / "images" / split).glob("*.png"))
        self.split = split
        self.off = off

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        p = self.imgs[i]
        img = Image.open(p).convert("RGB")
        bx = []
        lb = []
        for ln in (
            (YD / "labels" / self.split / (p.stem + ".txt")).read_text().splitlines()
        ):
            if not ln.strip():
                continue
            m, cx, cy, nw, nh = ln.split()
            m = int(m)
            cx, cy, nw, nh = map(float, (cx, cy, nw, nh))
            bx.append(
                [
                    (cx - nw / 2) * W,
                    (cy - nh / 2) * H,
                    (cx + nw / 2) * W,
                    (cy + nh / 2) * H,
                ]
            )
            lb.append(m + self.off)
        return TF.to_tensor(img), {
            "boxes": torch.tensor(bx, dtype=torch.float32),
            "labels": torch.tensor(lb, dtype=torch.int64),
        }


def collate(b):
    return tuple(zip(*b))


def build(name):
    if name == "FasterRCNN-R50":
        m = fasterrcnn_resnet50_fpn(weights="DEFAULT")
        inf = m.roi_heads.box_predictor.cls_score.in_features
        m.roi_heads.box_predictor = FastRCNNPredictor(inf, NC + 1)
        off = 1
    elif name == "RetinaNet-R50":
        m = retinanet_resnet50_fpn(weights="DEFAULT")
        m.head.classification_head = RetinaNetClassificationHead(
            256, m.head.classification_head.num_anchors, NC
        )
        off = 0
    else:  # FCOS-R50
        m = fcos_resnet50_fpn(weights="DEFAULT")
        m.head.classification_head = FCOSClassificationHead(256, 1, NC)
        off = 0
    m.transform.min_size = (512,)
    m.transform.max_size = 800
    return m, off


def evaluate(model, off):
    model.eval()
    dts = []
    ds = DS("val", off)
    with torch.no_grad():
        for i in range(len(ds)):
            img, _ = ds[i]
            iid = nid[ds.imgs[i].name]
            out = model([img.to(DEV)])[0]
            for box, lb, sc in zip(
                out["boxes"].cpu().numpy(),
                out["labels"].cpu().numpy(),
                out["scores"].cpu().numpy(),
            ):
                if sc < 0.001:
                    continue
                idx = int(lb) - off
                if idx not in m2c:
                    continue
                x1, y1, x2, y2 = box
                dts.append(
                    {
                        "image_id": iid,
                        "category_id": m2c[idx],
                        "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                        "score": float(sc),
                    }
                )
    if not dts:
        return 0.0, 0.0, 0
    e = COCOeval(cocoGt, cocoGt.loadRes(dts), "bbox")
    e.params.iouThrs = np.linspace(0.75, 0.95, 5)
    e.evaluate()
    e.accumulate()
    e.summarize()
    e2 = COCOeval(cocoGt, cocoGt.loadRes(dts), "bbox")
    e2.params.iouThrs = np.array([0.75])
    e2.evaluate()
    e2.accumulate()
    e2.summarize()
    return float(e.stats[0]), float(e2.stats[0]), len(dts)


resf = RUNS / "baselines.json"
results = json.loads(resf.read_text()) if resf.exists() else []
have = {r["model"] for r in results}

for name in ["FasterRCNN-R50", "RetinaNet-R50", "FCOS-R50"]:
    if name in have:
        continue
    print(f"\n===== TRAIN {name} =====", flush=True)
    model, off = build(name)
    model.to(DEV)
    dl = DataLoader(
        DS("train", off), batch_size=2, shuffle=True, collate_fn=collate, num_workers=0
    )
    opt = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=0.005,
        momentum=0.9,
        weight_decay=5e-4,
    )
    sch = torch.optim.lr_scheduler.StepLR(opt, step_size=16, gamma=0.1)
    EP = 20
    for ep in range(EP):
        model.train()
        tot = 0.0
        for imgs, tgts in dl:
            imgs = [x.to(DEV) for x in imgs]
            tgts = [{k: v.to(DEV) for k, v in t.items()} for t in tgts]
            loss = sum(model(imgs, tgts).values())
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss)
        sch.step()
        if (ep + 1) % 5 == 0:
            print(f"  {name} ep{ep + 1}/{EP} loss={tot / len(dl):.3f}", flush=True)
    pm = sum(p.numel() for p in model.parameters()) / 1e6
    mAP, ap75, ndet = evaluate(model, off)
    results.append(
        {
            "model": name,
            "params_M": round(pm, 1),
            "mAP_75_95": round(mAP, 4),
            "ap75": round(ap75, 4),
            "epochs": EP,
            "imgsz": "512(min)",
            "val_det": ndet,
        }
    )
    resf.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(
        f">>> {name}: mAP@[0.75:0.95]={mAP:.4f} AP@0.75={ap75:.4f} params={pm:.1f}M",
        flush=True,
    )

print("\n===== 베이스라인 비교표 (fold0 val) =====")
print(f"{'model':<16}{'params_M':>9}{'mAP@.75:.95':>13}{'AP@.75':>9}")
for r in sorted(results, key=lambda x: -x["mAP_75_95"]):
    print(
        f"{r['model']:<16}{r['params_M']:>9}{r['mAP_75_95']:>13}{r.get('ap75', 0):>9}"
    )
