"""GT 누락 교정 (로컬 corrected 라벨). 대회 원본·팀 파일 무수정 — 우리 사본만 생성.
유형 A(약품 통째 누락) = combo파일명 약품 − GT클래스 차집합으로 결정론적 완전탐지 → 검출기가 박스 채움.
유형 B(동일클래스 2번째 실알약) = 검출박스 중 모든 GT와 IoU<0.3인 여분 → 플래그만(클래스 수동확인 필요).
산출: data/gt_corrected/{corrections.json, corrected_coco.json, review/*.png, report.md}"""

import json
import re
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
SSOT = LHK / "data/processed"
OUT = LHK / "data/gt_corrected"
REVIEW = OUT / "review"
REVIEW.mkdir(parents=True, exist_ok=True)
CONF = 0.25

cm = json.load(open(SSOT / "class_map.json"))
m2c = {int(k): v for k, v in cm["model_index_to_category_id"].items()}
cats = json.load(open(LHK / "handoff_realcopy/target_categories_schema.json"))
name_of = {c["id"]: c.get("name", str(c["id"])) for c in cats}

# 이미지별 GT (category_id, [x,y,w,h])
img_boxes, img_wh = defaultdict(list), {}
for jf in TA.rglob("*.json"):
    d = json.load(open(jf))
    im = d["images"][0]
    fn = im["file_name"]
    img_wh[fn] = (im["width"], im["height"])
    for a in d["annotations"]:
        img_boxes[fn].append((int(a["category_id"]), [float(v) for v in a["bbox"]]))
files = sorted(img_boxes)


def combo_cats(fn):
    combo = fn.split("_")[0]  # K-001900-016548-...
    return [int(x) for x in re.findall(r"\d+", combo)]


def iou_xywh(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    u = aw * ah + bw * bh - inter
    return inter / u if u > 0 else 0.0


det = YOLO(str(DET))
corrections = {}  # fn -> [ {category_id, bbox, conf} ]
typeB_flags = {}  # fn -> [ {det_class, bbox, conf} ]
unresolved = []  # 검출기가 못 채운 gap

for fn in files:
    gt_cats = [c for c, _ in img_boxes[fn]]
    gt_bxs = [b for _, b in img_boxes[fn]]
    missing = [c for c in combo_cats(fn) if c not in gt_cats]  # 유형 A gap
    if not missing:
        continue
    r = det.predict(str(TI / fn), conf=CONF, imgsz=640, device="mps", verbose=False)[0]
    dets = []
    for b, cl, s in zip(
        r.boxes.xyxy.cpu().numpy(),
        r.boxes.cls.cpu().numpy(),
        r.boxes.conf.cpu().numpy(),
    ):
        x1, y1, x2, y2 = b
        dets.append(
            (
                m2c[int(cl)],
                [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                float(s),
            )
        )
    added = []
    for mc in missing:
        # 그 클래스 검출 중, 기존 GT와 안 겹치는(IoU<0.3) 최고 conf
        cand = [
            d
            for d in dets
            if d[0] == mc and all(iou_xywh(d[1], g) < 0.3 for g in gt_bxs)
        ]
        cand.sort(key=lambda d: -d[2])
        if cand:
            added.append(
                {
                    "category_id": mc,
                    "bbox": [round(v, 1) for v in cand[0][1]],
                    "conf": round(cand[0][2], 3),
                }
            )
        else:
            unresolved.append((fn, mc))
    if added:
        corrections[fn] = added

# 유형 B: 어떤 GT와도 IoU<0.3인 여분 검출(누락 아닌 이미지 포함) — 클래스 신뢰낮음, 플래그만
for fn in files:
    gt_bxs = [b for _, b in img_boxes[fn]]
    already = corrections.get(fn, [])
    r = det.predict(str(TI / fn), conf=0.35, imgsz=640, device="mps", verbose=False)[0]
    for b, cl, s in zip(
        r.boxes.xyxy.cpu().numpy(),
        r.boxes.cls.cpu().numpy(),
        r.boxes.conf.cpu().numpy(),
    ):
        x1, y1, x2, y2 = b
        bb = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
        if all(iou_xywh(bb, g) < 0.3 for g in gt_bxs) and all(
            iou_xywh(bb, a["bbox"]) < 0.3 for a in already
        ):
            typeB_flags.setdefault(fn, []).append(
                {
                    "det_class": m2c[int(cl)],
                    "bbox": [round(v, 1) for v in bb],
                    "conf": round(float(s), 3),
                }
            )

# 저장: corrections.json
json.dump(
    {
        "type_A_autofill": corrections,
        "type_B_manual_check": typeB_flags,
        "unresolved_gaps": unresolved,
    },
    open(OUT / "corrections.json", "w"),
    ensure_ascii=False,
    indent=1,
)

# corrected COCO (원본 + A 추가). image_id = 파일명 숫자? 여기선 순번.
coco = {
    "images": [],
    "annotations": [],
    "categories": [
        {"id": c, "name": name_of.get(c, str(c))} for c in sorted(set(m2c.values()))
    ],
}
aid = 1
for i, fn in enumerate(files, 1):
    w, h = img_wh[fn]
    coco["images"].append({"id": i, "file_name": fn, "width": w, "height": h})
    for c, b in img_boxes[fn]:
        coco["annotations"].append(
            {
                "id": aid,
                "image_id": i,
                "category_id": c,
                "bbox": b,
                "area": b[2] * b[3],
                "iscrowd": 0,
                "source": "official",
            }
        )
        aid += 1
    for a in corrections.get(fn, []):
        b = a["bbox"]
        coco["annotations"].append(
            {
                "id": aid,
                "image_id": i,
                "category_id": a["category_id"],
                "bbox": b,
                "area": b[2] * b[3],
                "iscrowd": 0,
                "source": "corrected_A",
            }
        )
        aid += 1
json.dump(coco, open(OUT / "corrected_coco.json", "w"), ensure_ascii=False)

# 검수 오버레이 (초록=원본 GT, 빨강=추가된 교정박스 + 클래스명)
for fn in corrections:
    im = cv2.imread(str(TI / fn))
    for c, (x, y, w, h) in img_boxes[fn]:
        cv2.rectangle(im, (int(x), int(y)), (int(x + w), int(y + h)), (0, 200, 0), 3)
    for a in corrections[fn]:
        x, y, w, h = a["bbox"]
        cv2.rectangle(im, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 4)
        cv2.putText(
            im,
            f"+{a['category_id']} {name_of.get(a['category_id'], '')} {a['conf']}",
            (int(x), max(20, int(y) - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )
    cv2.imwrite(str(REVIEW / f"corr_{fn}"), im)

# 컨택트시트
ps = sorted(REVIEW.glob("corr_*.png"))
if ps:
    cols = 4
    rows = (len(ps) + cols - 1) // cols
    tw, th = 300, 394
    canvas = np.full((rows * th, cols * tw, 3), 30, np.uint8)
    for i, p in enumerate(ps):
        rr, cc = divmod(i, cols)
        canvas[rr * th : (rr + 1) * th, cc * tw : (cc + 1) * tw] = cv2.resize(
            cv2.imread(str(p)), (tw, th)
        )
    cv2.imwrite(str(OUT / "review_contact.png"), canvas)

# report.md
n_boxes = sum(len(v) for v in corrections.values())
lines = [
    "# GT corrected 라벨 리포트 (유형 A 자동채움)",
    "",
    f"- 교정 이미지: **{len(corrections)}장**, 추가 박스: **{n_boxes}개** (유형 A, combo-차집합 확정)",
    f"- 미해결 gap(검출기 못 채움, 수동 필요): {len(unresolved)}건 {unresolved}",
    f"- 유형 B 후보(동일클래스 2번째 실알약 의심, 클래스 수동확인): {sum(len(v) for v in typeB_flags.values())}건 / {len(typeB_flags)}장",
    "",
    "## 유형 A 추가 박스",
    "| 이미지 | +클래스 | 약품명 | conf |",
    "|---|---|---|---|",
]
for fn, adds in corrections.items():
    for a in adds:
        lines.append(
            f"| {fn[:44]} | {a['category_id']} | {name_of.get(a['category_id'], '')} | {a['conf']} |"
        )
(OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")

print(
    f"유형A 교정: {len(corrections)}장 / {n_boxes}박스 | 미해결 {len(unresolved)} | 유형B후보 {sum(len(v) for v in typeB_flags.values())}건"
)
print(
    f"산출: {OUT}/ (corrections.json, corrected_coco.json, review/, review_contact.png, report.md)"
)
