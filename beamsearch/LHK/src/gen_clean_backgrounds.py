"""알약-free 배경 패치 생성 (384×384 클린 타일). 견고한 마스크:
 풀이미지에 학습된 YOLO 검출기(학습분포=신뢰) 실행 → 검출박스 ∪ GT박스 = 완전 알약마스크 → 그걸 피해 크롭.
 (GT 단독은 ~30% 알약 누락 확인 / 패치단위 검출은 스케일 불일치로 실패 → 풀이미지 검출이 정답.)
소스 = fold0 train만(val/test 금지). 블루그레이 도메인 + 원본 밝기·비네팅 자연 보존.
976×1280 풀프레임 클린은 알약 산개로 자동추출 불가 → 큰 패치로 이음새 최소화, Codex가 타일링."""

import json
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
from ultralytics import YOLO

BASE = Path("/Users/macbook/dev/learning/codeit/01_Proj_HealthEat_cla")
TA = BASE / "01_data/01_sprint_ai_project1_data/train_annotations"
TI = BASE / "01_data/01_sprint_ai_project1_data/train_images"
LHK = BASE / "project1-3team/beamsearch/LHK"
DET = LHK / "runs/e2_synth/weights/best.pt"
OUT = Path.home() / "Desktop/healtheat_backgrounds_16/autoclean_pillfree"
P = 384
PAD = 26
TARGET = 64
PER_IMG = 3
DET_CONF = 0.10  # 낮게 → 흐릿한 알약까지 포착(배경 오염 방지 우선)

img_boxes, img_anns = defaultdict(list), defaultdict(list)
for jf in TA.rglob("*.json"):
    d = json.load(open(jf))
    fn = d["images"][0]["file_name"]
    for a in d["annotations"]:
        img_boxes[fn].append([int(v) for v in a["bbox"]])
        img_anns[fn].append(int(a["category_id"]))
files = sorted(img_anns)
sets = {fn: frozenset(img_anns[fn]) for fn in files}
combo_of = {c: i for i, c in enumerate(dict.fromkeys(sets[fn] for fn in files))}
groups = np.array([combo_of[sets[fn]] for fn in files])
rng = np.random.default_rng(42)
uniq = np.array(sorted(set(groups)))
rng.shuffle(uniq)
fold_groups = np.array_split(uniq, 5)
g2f = {int(g): k for k, fl in enumerate(fold_groups) for g in fl}
train0 = [fn for i, fn in enumerate(files) if g2f[int(groups[i])] != 0]
rng.shuffle(train0)

det = YOLO(str(DET))


def pill_mask_boxes(fn, im_bgr):
    """GT ∪ 검출기(풀이미지) 박스 = 완전 알약 박스 리스트 [x,y,w,h]."""
    boxes = [list(b) for b in img_boxes[fn]]
    r = det.predict(im_bgr, conf=DET_CONF, imgsz=640, device="mps", verbose=False)[0]
    for x1, y1, x2, y2 in r.boxes.xyxy.cpu().numpy():
        boxes.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])
    return boxes


def clean_patches(fn, want, rng):
    im_bgr = cv2.imread(str(TI / fn))
    im = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
    ih, iw = im.shape[:2]
    boxes = pill_mask_boxes(fn, im_bgr)
    out = []
    for _ in range(400):
        if len(out) >= want:
            break
        x = int(rng.integers(0, iw - P))
        y = int(rng.integers(0, ih - P))
        if not all(
            x + P <= bx - PAD
            or x >= bx + bw + PAD
            or y + P <= by - PAD
            or y >= by + bh + PAD
            for bx, by, bw, bh in boxes
        ):
            continue
        patch = im[y : y + P, x : x + P]
        pm = patch.reshape(-1, 3).mean(0)
        if pm[2] > pm[0] and 70 < pm.mean() < 185:
            out.append(patch)
    return out


kept = 0
OUT.mkdir(parents=True, exist_ok=True)
for old in OUT.glob("*.png"):
    old.unlink()
for fn in train0:
    if kept >= TARGET:
        break
    for patch in clean_patches(fn, PER_IMG, np.random.default_rng(hash(fn) % 2**32)):
        if kept >= TARGET:
            break
        cv2.imwrite(
            str(OUT / f"bg_clean_{kept:02d}.png"),
            cv2.cvtColor(patch, cv2.COLOR_RGB2BGR),
        )
        kept += 1

print(f"생성 {kept}장 (384×384, GT∪검출기 마스크로 알약 완전 배제)")
ps = sorted(OUT.glob("bg_clean_*.png"))
if ps:
    cols, th = 8, 110
    rows = (len(ps) + cols - 1) // cols
    canvas = np.full((rows * th, cols * th, 3), 40, np.uint8)
    for i, p in enumerate(ps):
        rr, cc = divmod(i, cols)
        canvas[rr * th : (rr + 1) * th, cc * th : (cc + 1) * th] = cv2.resize(
            cv2.imread(str(p)), (th, th)
        )
    cv2.imwrite(
        "/private/tmp/claude-501/-Users-macbook-dev-learning-codeit/473e8dff-50a9-4f8c-9fe2-897163954e84/scratchpad/final_bg_contact.png",
        canvas,
    )
    print(f"컨택트시트 저장 ({len(ps)}장)")
