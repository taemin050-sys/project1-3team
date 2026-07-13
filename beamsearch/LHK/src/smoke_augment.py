"""증강 엔진 스모크: (2) 없이 (1) 데이터로 검증.
(1) 알약을 bbox 크롭→Otsu 분할→WB정규화→컷아웃 → (1) 배경에 비겹침 합성 → 시각화 저장."""

import sys
import json
import random
from pathlib import Path
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).parent))
import augment as A

LHK = Path(
    "/Users/macbook/dev/learning/codeit/01_Proj_HealthEat_cla/project1-3team/beamsearch/LHK"
)
YD = LHK / "data/yolo"
W, H = 976, 1280
m2c = {
    int(k): v
    for k, v in json.load(open(LHK / "data/processed/class_map.json"))[
        "model_index_to_category_id"
    ].items()
}
rng = random.Random(42)
train_imgs = sorted((YD / "images/train").glob("*.png"))

# 1) 컷아웃 풀: (1) 알약 크롭 → 분할 → WB정규화 → RGBA
pool = []
for p in rng.sample(train_imgs, 12):
    img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
    for ln in (YD / "labels/train" / (p.stem + ".txt")).read_text().splitlines():
        if not ln.strip():
            continue
        m, cx, cy, nw, nh = ln.split()
        m = int(m)
        cx, cy, nw, nh = map(float, (cx, cy, nw, nh))
        x, y, w, h = (
            int((cx - nw / 2) * W),
            int((cy - nh / 2) * H),
            int(nw * W),
            int(nh * H),
        )
        pad = 8
        crop = img[max(0, y - pad) : y + h + pad, max(0, x - pad) : x + w + pad]
        if crop.size == 0:
            continue
        mask = A.segment_pill(crop)
        rgba = A.cutout(crop, mask, erode_px=2) if mask is not None else None
        if rgba is not None and min(rgba.shape[:2]) > 12:
            rgba[:, :, :3] = A.wb_gray_world(
                rgba[:, :, :3], rgba[:, :, 3]
            )  # WB 정규화 데모
            pool.append((rgba, m2c[m]))
print(f"컷아웃 풀: {len(pool)}개 (분할·WB정규화 완료)")


# 2) 합성 4장 + 검증
def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix, iy = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, ix2 - ix) * max(0, iy2 - iy)
    return inter / (aw * ah + bw * bh - inter + 1e-6)


vizs = []
for i in range(4):
    bg = A.sample_background(train_imgs, size=(W, H), rng=rng)
    comp, anns = A.compose(bg, pool, min_n=2, max_n=4, rng=rng)
    boxes = [b for _, b in anns]
    valid = all(
        bw > 0 and bh > 0 and bx >= 0 and by >= 0 and bx + bw <= W and by + bh <= H
        for _, (bx, by, bw, bh) in [(0, b) for b in boxes]
    )
    max_iou = max(
        [
            iou(boxes[a], boxes[c])
            for a in range(len(boxes))
            for c in range(a + 1, len(boxes))
        ]
        or [0]
    )
    print(
        f"  합성#{i}: {len(anns)}알 | bbox유효={valid} | 최대겹침IoU={max_iou:.3f} | 크기={comp.shape[1]}x{comp.shape[0]}"
    )
    v = comp.copy()
    for cat, (bx, by, bw, bh) in anns:
        cv2.rectangle(v, (bx, by), (bx + bw, by + bh), (0, 255, 0), 3)
        cv2.putText(
            v,
            str(cat),
            (bx, max(0, by - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
        )
    vizs.append(cv2.resize(v, (W // 2, H // 2)))

grid = np.vstack([np.hstack(vizs[:2]), np.hstack(vizs[2:])])
out = LHK / "runs/aug_smoke.png"
cv2.imwrite(str(out), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
print(f"\n시각화 저장: {out}  (2x2, bbox=초록)")
