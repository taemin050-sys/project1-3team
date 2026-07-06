import json
from pathlib import Path

nbp = Path(
    "/Users/macbook/dev/learning/codeit/01_Proj_HealthEat_cla/project1-3team/beamsearch/LHK/03_cuda_baselines_colab.ipynb"
)
nb = json.loads(nbp.read_text(encoding="utf-8"))


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s}


def code(s):
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": s,
    }


frcnn = [
    md("""## 1b. Faster R-CNN (torchvision, warmup) — MPS서 RoIAlign CPU폴백으로 너무 느려(에폭 52분) CUDA로 이관
발산 방지: **선형 warmup + grad-clip**. (MPS 표의 0.0은 warmup 부재 발산 → 여기서 진짜 수치 산출)"""),
    code("""import torch, time
from PIL import Image
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
Wp, Hp = 976, 1280

class DS(Dataset):
    def __init__(self, sp): self.im = sorted((ROOT/"images"/sp).glob("*.png")); self.sp = sp
    def __len__(self): return len(self.im)
    def __getitem__(self, i):
        p = self.im[i]; img = Image.open(p).convert("RGB"); bx = []; lb = []
        for ln in (ROOT/"labels"/self.sp/(p.stem+".txt")).read_text().splitlines():
            if not ln.strip(): continue
            m, cx, cy, nw, nh = ln.split(); m = int(m); cx, cy, nw, nh = map(float, (cx, cy, nw, nh))
            bx.append([(cx-nw/2)*Wp, (cy-nh/2)*Hp, (cx+nw/2)*Wp, (cy+nh/2)*Hp]); lb.append(m+1)
        return TF.to_tensor(img), {"boxes": torch.tensor(bx, dtype=torch.float32), "labels": torch.tensor(lb, dtype=torch.int64)}
def coll(b): return tuple(zip(*b))

mdl = fasterrcnn_resnet50_fpn(weights="DEFAULT")
mdl.roi_heads.box_predictor = FastRCNNPredictor(mdl.roi_heads.box_predictor.cls_score.in_features, 57)
mdl.transform.min_size = (512,); mdl.transform.max_size = 800; mdl.cuda()
dl = DataLoader(DS("train"), batch_size=4, shuffle=True, collate_fn=coll, num_workers=2)
opt = torch.optim.SGD([p for p in mdl.parameters() if p.requires_grad], lr=0.005, momentum=0.9, weight_decay=5e-4)
BL, EP = 0.005, 24; wu = min(500, len(dl)*2); g = 0; t0 = time.time()
for ep in range(EP):
    mdl.train(); tot = 0.0
    for im, tg in dl:
        g += 1
        if g <= wu:
            for pg in opt.param_groups: pg["lr"] = BL*(0.01+0.99*g/wu)
        im = [x.cuda() for x in im]; tg = [{k: v.cuda() for k, v in t.items()} for t in tg]
        loss = sum(mdl(im, tg).values())
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(mdl.parameters(), 10); opt.step(); tot += float(loss)
    if ep >= int(EP*0.8):
        for pg in opt.param_groups: pg["lr"] = BL*0.1
    print(f"ep{ep+1}/{EP} loss={tot/len(dl):.3f} ({time.time()-t0:.0f}s)")

mdl.eval(); dts = []
with torch.no_grad():
    for p in val_imgs:
        o = mdl([TF.to_tensor(Image.open(p).convert("RGB")).cuda()])[0]
        for b, l, sc in zip(o["boxes"].cpu().numpy(), o["labels"].cpu().numpy(), o["scores"].cpu().numpy()):
            if sc < 0.001: continue
            idx = int(l) - 1
            if idx not in m2c: continue
            x1, y1, x2, y2 = b
            dts.append({"image_id": nid[p.name], "category_id": m2c[idx], "bbox": [float(x1), float(y1), float(x2-x1), float(y2-y1)], "score": float(sc)})
score("FasterRCNN-R50(warmup)", dts, sum(q.numel() for q in mdl.parameters())/1e6)"""),
]

# "## 2. mmdetection" 마크다운 셀 앞에 삽입
out = []
inserted = False
for c in nb["cells"]:
    src = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
    if (
        not inserted
        and c["cell_type"] == "markdown"
        and src.startswith("## 2. mmdetection")
    ):
        out.extend(frcnn)
        inserted = True
    out.append(c)
nb["cells"] = out
nbp.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("inserted FasterRCNN section:", inserted, "| total cells:", len(nb["cells"]))
