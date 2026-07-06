"""FasterRCNN 재실행 (LHK/MPS): warmup + grad-clip + NaN 감시로 발산 해결. baselines.json의 0.0 항목 교체."""

import json
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

DEV = "mps"
W, H = 976, 1280
NC = 56
LHK = Path(
    "/Users/macbook/dev/learning/codeit/01_Proj_HealthEat_cla/project1-3team/beamsearch/LHK"
)
YD, RUNS, SSOT = LHK / "data/yolo", LHK / "runs", LHK / "data/processed"
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
    def __init__(self, sp, off):
        self.imgs = sorted((YD / "images" / sp).glob("*.png"))
        self.sp = sp
        self.off = off

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        p = self.imgs[i]
        img = Image.open(p).convert("RGB")
        bx = []
        lb = []
        for ln in (
            (YD / "labels" / self.sp / (p.stem + ".txt")).read_text().splitlines()
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


model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
model.roi_heads.box_predictor = FastRCNNPredictor(
    model.roi_heads.box_predictor.cls_score.in_features, NC + 1
)
model.transform.min_size = (512,)
model.transform.max_size = 800
model.to(DEV)
dl = DataLoader(
    DS("train", 1), batch_size=2, shuffle=True, collate_fn=collate, num_workers=0
)

BASE_LR, EP = 0.005, 24
opt = torch.optim.SGD(
    [p for p in model.parameters() if p.requires_grad],
    lr=BASE_LR,
    momentum=0.9,
    weight_decay=5e-4,
)
warmup_iters = min(500, len(dl) * 2)
git = 0
t0 = time.time()
for ep in range(EP):
    model.train()
    tot = 0.0
    for imgs, tgts in dl:
        git += 1
        if git <= warmup_iters:  # 선형 warmup
            for g in opt.param_groups:
                g["lr"] = BASE_LR * (0.01 + 0.99 * git / warmup_iters)
        imgs = [x.to(DEV) for x in imgs]
        tgts = [{k: v.to(DEV) for k, v in t.items()} for t in tgts]
        loss = sum(model(imgs, tgts).values())
        if not torch.isfinite(loss):  # NaN 감시
            print(f"!! NaN/inf loss at ep{ep} iter{git} — skip")
            opt.zero_grad()
            continue
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)  # grad-clip
        opt.step()
        tot += float(loss.detach())
    if ep >= int(EP * 0.8):
        for g in opt.param_groups:
            g["lr"] = BASE_LR * 0.1  # 후반 decay
    print(
        f"ep{ep + 1}/{EP} loss={tot / len(dl):.3f} lr={opt.param_groups[0]['lr']:.5f} ({time.time() - t0:.0f}s)",
        flush=True,
    )

# eval
model.eval()
dts = []
vds = DS("val", 1)
with torch.no_grad():
    for i in range(len(vds)):
        img, _ = vds[i]
        out = model([img.to(DEV)])[0]
        for box, lb, sc in zip(
            out["boxes"].cpu().numpy(),
            out["labels"].cpu().numpy(),
            out["scores"].cpu().numpy(),
        ):
            if sc < 0.001:
                continue
            idx = int(lb) - 1
            if idx not in m2c:
                continue
            x1, y1, x2, y2 = box
            dts.append(
                {
                    "image_id": nid[vds.imgs[i].name],
                    "category_id": m2c[idx],
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "score": float(sc),
                }
            )
mAP = ap75 = 0.0
if dts:
    e = COCOeval(cocoGt, cocoGt.loadRes(dts), "bbox")
    e.params.iouThrs = np.linspace(0.75, 0.95, 5)
    e.evaluate()
    e.accumulate()
    e.summarize()
    mAP = float(e.stats[0])
    e2 = COCOeval(cocoGt, cocoGt.loadRes(dts), "bbox")
    e2.params.iouThrs = np.array([0.75])
    e2.evaluate()
    e2.accumulate()
    e2.summarize()
    ap75 = float(e2.stats[0])

resf = RUNS / "baselines.json"
res = json.loads(resf.read_text())
res = [r for r in res if r["model"] != "FasterRCNN-R50"]  # 기존 0.0 교체
res.append(
    {
        "model": "FasterRCNN-R50",
        "params_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
        "mAP_75_95": round(mAP, 4),
        "ap75": round(ap75, 4),
        "epochs": EP,
        "imgsz": "512(min)+warmup",
        "val_det": len(dts),
    }
)
resf.write_text(json.dumps(res, ensure_ascii=False, indent=2))
print(
    f"\n>>> FasterRCNN-R50 (warmup): mAP@[0.75:0.95]={mAP:.4f} AP@0.75={ap75:.4f} val_det={len(dts)}"
)
