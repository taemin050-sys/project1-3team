import json
from pathlib import Path

LHK = Path(
    "/Users/macbook/dev/learning/codeit/01_Proj_HealthEat_cla/project1-3team/beamsearch/LHK"
)


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


cells = []
cells.append(
    md("""# 🖥️ CUDA 베이스라인 (Colab/Runpod) — RT-DETR · Cascade R-CNN · DINO

MPS로 학습 불가한 모델을 **동일 fold0 번들**로 CUDA에서 돌린다. MPS 5종과 **같은 mAP@[0.75:0.95] 하니스**를 써서 결과를 그대로 합류.

## 실행 방법
1. **런타임 → GPU** 로 변경(Colab: 런타임 유형 변경 → T4/GPU).
2. `lhk_cuda_bundle.zip` 을 업로드(왼쪽 파일창 드래그) — 또는 Google Drive에 두고 마운트.
3. 위에서부터 셀 실행. **RT-DETR 먼저**(가장 안정) → mmdet(Cascade/DINO)는 설치 버전 이슈가 있을 수 있어 best-effort.
4. 완료 후 `cuda_baselines.json` 을 다운로드해 KAI에게 전달 → MPS 표에 합류.""")
)

cells.append(md("## 0. 셋업 · 번들 언집 · 공통 하니스"))
cells.append(
    code("""import os, zipfile
from pathlib import Path
import torch
BUNDLE_ZIP = "lhk_cuda_bundle.zip"        # 업로드한 파일명 (Drive면 경로 수정)
ROOT = Path("/content/lhk_cuda_bundle")
if not ROOT.exists():
    with zipfile.ZipFile(BUNDLE_ZIP) as z: z.extractall("/content")
print("CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "(GPU 런타임으로 변경 필요!)")
# data.yaml 절대경로 보정 (ultralytics용)
dy = (ROOT/"data.yaml").read_text()
if "path: ." in dy: (ROOT/"data.yaml").write_text(dy.replace("path: .", f"path: {ROOT}"))
print("bundle:", sorted(p.name for p in ROOT.iterdir()))""")
)

cells.append(
    code("""# 공통 mAP@[0.75:0.95] 하니스 (MPS와 동일) — 예측 dts는 category_id=dl_idx, image_id=val 정렬순 1..51
import json, numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
m2c = {int(k): v for k, v in json.load(open(ROOT/"class_map.json"))["model_index_to_category_id"].items()}
cocoGt = COCO(str(ROOT/"val_gt.json"))
val_imgs = sorted((ROOT/"images/val").glob("*.png"))
nid = {p.name: i+1 for i, p in enumerate(val_imgs)}
RESULTS = []
def score(name, dts, params_M):
    if not dts:
        mAP = ap75 = 0.0
    else:
        e = COCOeval(cocoGt, cocoGt.loadRes(dts), "bbox"); e.params.iouThrs = np.linspace(0.75, 0.95, 5)
        e.evaluate(); e.accumulate(); e.summarize(); mAP = float(e.stats[0])
        e2 = COCOeval(cocoGt, cocoGt.loadRes(dts), "bbox"); e2.params.iouThrs = np.array([0.75])
        e2.evaluate(); e2.accumulate(); e2.summarize(); ap75 = float(e2.stats[0])
    RESULTS.append({"model": name, "params_M": round(params_M, 1), "mAP_75_95": round(mAP, 4), "ap75": round(ap75, 4), "val_det": len(dts)})
    json.dump(RESULTS, open(ROOT/"cuda_baselines.json", "w"), ensure_ascii=False, indent=2)
    print(f">>> {name}: mAP@[0.75:0.95]={mAP:.4f}  AP@0.75={ap75:.4f}  val_det={len(dts)}")""")
)

cells.append(md("## 1. RT-DETR (ultralytics) — 안정적, 먼저 실행"))
cells.append(
    code("""!pip -q install ultralytics
from ultralytics import RTDETR
m = RTDETR("rtdetr-l.pt")
m.train(data=str(ROOT/"data.yaml"), epochs=50, imgsz=640, batch=8, device=0, seed=42,
        project=str(ROOT/"runs"), name="rtdetr", exist_ok=True, plots=False, verbose=False)
best = RTDETR(str(ROOT/"runs/rtdetr/weights/best.pt"))
dts = []
for p in val_imgs:
    r = best.predict(str(p), conf=0.001, iou=0.6, max_det=20, device=0, verbose=False)[0]
    for b, c, s in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy()):
        x1, y1, x2, y2 = b
        dts.append({"image_id": nid[p.name], "category_id": m2c[int(c)], "bbox": [float(x1), float(y1), float(x2-x1), float(y2-y1)], "score": float(s)})
score("RT-DETR-l", dts, sum(pp.numel() for pp in best.model.parameters())/1e6)""")
)

cells.append(
    md("""## 2. mmdetection 설치 (Cascade · DINO 공용) — ⚠️ best-effort
Colab의 torch/CUDA에 맞춰 `mim`이 mmcv를 매칭. 실패 시 버전 힌트를 KAI에게 전달해 함께 조정.""")
)
cells.append(
    code("""!pip -q install -U openmim
!mim install -q "mmengine>=0.10.0" "mmcv>=2.0.0,<2.2.0" "mmdet>=3.2.0"
import mmdet, mmcv, mmengine; print("mmdet", mmdet.__version__, "| mmcv", mmcv.__version__)""")
)

cells.append(
    code("""# mmdet 학습 헬퍼: base config를 우리 fold0 COCO(56클래스)로 override → 학습 → val 추론 → 하니스 채점
from mmengine.config import Config
from mmengine.runner import Runner

CLASSES = tuple(str(m2c[i]) for i in range(56))
META = dict(classes=CLASSES)

def set_num_classes(model_cfg, nc=56):
    rh = model_cfg.get("roi_head")
    if rh is not None:                       # Cascade/Faster: roi_head.bbox_head (list 또는 dict)
        bh = rh["bbox_head"]
        for h in (bh if isinstance(bh, list) else [bh]): h["num_classes"] = nc
    if model_cfg.get("bbox_head") is not None:   # DINO/DETR 계열
        model_cfg["bbox_head"]["num_classes"] = nc

def run_mmdet(cfg_path, run_name, epochs=24, base_lr=None):
    cfg = Config.fromfile(cfg_path)
    for split, ann, pref in [("train", "coco/train.json", "images/train/"), ("val", "coco/val.json", "images/val/")]:
        dl = cfg[f"{split}_dataloader"]
        dl["dataset"].update(data_root=str(ROOT), ann_file=ann, data_prefix=dict(img=pref), metainfo=META)
    cfg.train_dataloader["batch_size"] = 2
    cfg.test_dataloader = cfg.val_dataloader
    cfg.val_evaluator.update(ann_file=str(ROOT/"coco/val.json"), metric="bbox"); cfg.test_evaluator = cfg.val_evaluator
    set_num_classes(cfg.model, 56)
    cfg.train_cfg["max_epochs"] = epochs
    if base_lr is not None: cfg.optim_wrapper["optimizer"]["lr"] = base_lr
    cfg.default_hooks["checkpoint"].update(interval=epochs, save_best="coco/bbox_mAP")
    cfg.work_dir = str(ROOT/f"runs/{run_name}")
    cfg.load_from = None  # 필요시 COCO 사전학습 ckpt URL 지정
    runner = Runner.from_cfg(cfg); runner.train()
    return runner, cfg

def mmdet_eval(run_name, cfg, label):
    from mmdet.apis import init_detector, inference_detector
    import glob
    ck = sorted(glob.glob(str(ROOT/f"runs/{run_name}/best_*.pth")) or glob.glob(str(ROOT/f"runs/{run_name}/epoch_*.pth")))[-1]
    model = init_detector(cfg, ck, device="cuda:0")
    dts = []
    for p in val_imgs:
        res = inference_detector(model, str(p)).pred_instances
        bb = res.bboxes.cpu().numpy(); lb = res.labels.cpu().numpy(); sc = res.scores.cpu().numpy()
        for (x1, y1, x2, y2), l, s in zip(bb, lb, sc):
            if s < 0.001: continue
            dts.append({"image_id": nid[p.name], "category_id": m2c[int(l)], "bbox": [float(x1), float(y1), float(x2-x1), float(y2-y1)], "score": float(s)})
    pm = sum(q.numel() for q in model.parameters())/1e6
    score(label, dts, pm)""")
)

cells.append(md("## 3. Cascade R-CNN (고-IoU 정합 2-stage)"))
cells.append(
    code("""r, c = run_mmdet("mmdet::cascade_rcnn/cascade-rcnn_r50_fpn_1x_coco.py", "cascade", epochs=24)
mmdet_eval("cascade", c, "Cascade-RCNN-R50")""")
)

cells.append(md("## 4. DINO (정확도 트랜스포머) — 수렴 위해 에폭↑ 권장"))
cells.append(
    code("""r, c = run_mmdet("mmdet::dino/dino-4scale_r50_8xb2-12e_coco.py", "dino", epochs=36)
mmdet_eval("dino", c, "DINO-4scale-R50")""")
)

cells.append(
    md("""## 5. DINOv2-frozen 백본 (선택·심화)
`torch.hub.load('facebookresearch/dinov2','dinov2_vits14')` 를 frozen 특징추출기로 쓰고 경량 검출 헤드를 얹는 방식. 배선이 커서 시간 여유 시 별도 진행(여기선 생략).""")
)

cells.append(md("## 6. 결과 요약 · 내보내기"))
cells.append(
    code("""import pandas as pd
df = pd.DataFrame(RESULTS).sort_values("mAP_75_95", ascending=False)
print(df.to_string(index=False))
print("\\n→ /content/lhk_cuda_bundle/cuda_baselines.json 다운로드해서 KAI에게 전달")
from google.colab import files  # Colab 전용
files.download(str(ROOT/"cuda_baselines.json"))""")
)

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
out = LHK / "03_cuda_baselines_colab.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("written:", out, "| cells:", len(cells))
