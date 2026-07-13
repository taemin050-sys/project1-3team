"""CUDA 노트북용 자립형 번들 생성 (② 방식): 동일 fold0 split을 고정 패키징.
포함: images(train/val 실파일) · YOLO labels+data.yaml · COCO train/val.json · class_map · val_gt · README."""

import json
import shutil
import zipfile
from pathlib import Path

BASE = Path("/Users/macbook/dev/learning/codeit/01_Proj_HealthEat_cla")
LHK = BASE / "project1-3team/beamsearch/LHK"
YD, SSOT, RUNS = LHK / "data/yolo", LHK / "data/processed", LHK / "runs"
W, H = 976, 1280
B = BASE / "lhk_cuda_bundle"  # 저장소(project1-3team) 밖 → git 오염 없음
if B.exists():
    shutil.rmtree(B)
m2c = {
    int(k): v
    for k, v in json.load(open(SSOT / "class_map.json"))[
        "model_index_to_category_id"
    ].items()
}

# 1) 이미지(심링크→실파일 복사) + 라벨
for sp in ("train", "val"):
    (B / "images" / sp).mkdir(parents=True, exist_ok=True)
    (B / "labels" / sp).mkdir(parents=True, exist_ok=True)
    for p in sorted((YD / "images" / sp).glob("*.png")):
        shutil.copy(p, B / "images" / sp / p.name)  # copy는 심링크 타깃 실내용 복사
    for p in sorted((YD / "labels" / sp).glob("*.txt")):
        shutil.copy(p, B / "labels" / sp / p.name)

# 2) data.yaml (ultralytics RT-DETR용, 상대경로)
names = "\n".join(f"  {i}: '{m2c[i]}'" for i in range(56))
(B / "data.yaml").write_text(
    f"path: .\ntrain: images/train\nval: images/val\nnc: 56\nnames:\n{names}\n",
    encoding="utf-8",
)


# 3) COCO train/val.json (mmdet Cascade/DINO용; category_id=model_index 0..55)
def build_coco(sp):
    imgs = sorted((B / "images" / sp).glob("*.png"))
    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": str(m2c[i])} for i in range(56)],
    }
    aid = 1
    for iid, p in enumerate(imgs, 1):
        coco["images"].append({"id": iid, "file_name": p.name, "width": W, "height": H})
        for ln in (B / "labels" / sp / (p.stem + ".txt")).read_text().splitlines():
            if not ln.strip():
                continue
            m, cx, cy, nw, nh = ln.split()
            m = int(m)
            cx, cy, nw, nh = map(float, (cx, cy, nw, nh))
            x, y, w, h = (cx - nw / 2) * W, (cy - nh / 2) * H, nw * W, nh * H
            coco["annotations"].append(
                {
                    "id": aid,
                    "image_id": iid,
                    "category_id": m,
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                }
            )
            aid += 1
    return coco


(B / "coco").mkdir(exist_ok=True)
for sp in ("train", "val"):
    json.dump(build_coco(sp), open(B / "coco" / f"{sp}.json", "w"), ensure_ascii=False)

# 4) 하니스 자산 (MPS와 동일 지표 재현용): class_map, val_gt(dl_idx 기반)
shutil.copy(SSOT / "class_map.json", B / "class_map.json")
shutil.copy(RUNS / "val_gt.json", B / "val_gt.json")

# 5) README
(B / "README.md").write_text(
    """# LHK CUDA 베이스라인 번들 (fold0 고정)

MPS 베이스라인과 **동일한 fold0 split**(조합 단위 GroupKFold, seed=42). Colab/Runpod(CUDA)에서
RT-DETR·Cascade R-CNN·DINO/Co-DETR·DINOv2-frozen 베이스라인을 돌리기 위한 자립형 번들.

## 구조
- `images/{train,val}/*.png`  (train 181 / val 51, 976x1280)
- `labels/{train,val}/*.txt`  YOLO 포맷 (class=model_index 0..55)
- `data.yaml`                   ultralytics(RT-DETR)용
- `coco/{train,val}.json`     mmdet(Cascade/DINO)용 (category_id=model_index 0..55)
- `class_map.json`             model_index ↔ category_id(dl_idx) 매핑
- `val_gt.json`                ★ 최종 mAP 하니스 GT (category_id=dl_idx, image_id=val 정렬순 1..51)

## 공정 비교 규칙 (필수)
최종 지표는 **MPS와 동일 하니스**로: 예측 → model_index를 class_map으로 **dl_idx**로 변환 →
`val_gt.json` 대상 **pycocotools COCOeval, iouThrs=linspace(0.75,0.95,5)** 의 stats[0] = mAP@[0.75:0.95].
(coco/val.json은 학습중 val용; 최종 비교 지표는 반드시 val_gt.json으로 계산해 표에 append.)

## 대상 모델
- RT-DETR (ultralytics, data.yaml) — MPS 미지원(grid_sampler backward) → CUDA
- Cascade R-CNN (mmdet, coco/*.json) — 고-IoU 정합 2-stage
- DINO / Co-DETR (mmdet) — deformable-attention CUDA 전용
- DINOv2-frozen 백본 + 경량 헤드 (선택)
""",
    encoding="utf-8",
)

# 6) zip
zp = BASE / "lhk_cuda_bundle.zip"
if zp.exists():
    zp.unlink()
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
    for f in B.rglob("*"):
        if f.is_file():
            z.write(f, f.relative_to(B.parent))
mb = zp.stat().st_size / 1e6
ni = sum(1 for _ in B.rglob("*.png"))
print(f"번들: {zp}  ({mb:.1f} MB, 이미지 {ni}장)")
print("포함:", sorted(p.name for p in B.iterdir()))
