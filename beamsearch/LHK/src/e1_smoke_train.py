"""E1 베이스라인 스모크 학습 (LHK, MPS). 파이프라인 관통 검증용 — 초경량."""

from pathlib import Path
from ultralytics import YOLO

LHK = Path(
    "/Users/macbook/dev/learning/codeit/01_Proj_HealthEat_cla/project1-3team/beamsearch/LHK"
)
model = YOLO("yolo11n.pt")  # COCO 사전학습 nano (전이학습)
res = model.train(
    data=str(LHK / "data/yolo/data.yaml"),
    epochs=3,
    imgsz=640,
    batch=8,
    device="mps",  # LHK 정책: MacBook MPS
    seed=42,
    deterministic=True,
    workers=2,
    project=str(LHK / "runs"),
    name="e1_smoke",
    exist_ok=True,
    plots=False,
    verbose=True,
)
print("\n=== SMOKE DONE ===")
print("save_dir:", res.save_dir)
