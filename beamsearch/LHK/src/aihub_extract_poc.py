"""AI Hub '경구약제조합 5000종'(허가분 TS/TL 1,3-8)에서 우리 56클래스 이미지를 추출·변환 (POC).
- 누수0(대회=금지 조합, 허가는 무중복). 라벨: 드럭별 json(images[0].dl_idx=category_id, annotations[0].bbox).
- 비-우리 알약은 배경(연회색)으로 마스킹 → 라벨 완전, 미검출-혼동 방지.
- 희소클래스(대회<10) 포함 조합 우선, 캡으로 POC 규모 제한. 출력=COCO(kfold_exp EXP_SYNTH 호환)."""

import io
import json
import os
import sys
import zipfile
from pathlib import Path
from collections import defaultdict, Counter

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths

ALLOW = Path("/Volumes/USB 1T/사용 허가 조합/01.데이터/1.Training")
TS_DIR = ALLOW / "원천데이터/경구약제조합 5000종"
TL_DIR = ALLOW / "라벨링데이터/경구약제조합 5000종"
ZIPS = ["TS_1", "TS_3", "TS_4", "TS_5", "TS_6", "TS_7", "TS_8"]
OUT = paths.PROCESSED / "kaggle_aihub_poc"
CAP_COMBOS = 700  # POC 조합 수 (조합당 ~3장)
W, H = 976, 1280

cm = json.load(open(paths.SSOT / "class_map.json"))
c2m = {int(k): v for k, v in cm["category_id_to_model_index"].items()}
our = set(int(k) for k in c2m)  # our category_ids (dl_idx)
our_kc = set(f"{c:06d}" for c in our)

# 대회 희소클래스(현<10) — 우선순위
import json as J

cc = Counter()
for jf in paths.TRAIN_ANNOTATIONS.rglob("*.json"):
    for a in J.load(open(jf))["annotations"]:
        cc[int(a["category_id"])] += 1
rare_kc = set(f"{c:06d}" for c in our if cc[c] < 10)

# 1) combo → (ts_zip, tl_zip) 인덱스 + 이미지 멤버 목록
print("인덱스 구축 중...", flush=True)
combo_ts, combo_imgs = {}, defaultdict(list)
for z in ZIPS:
    zf = zipfile.ZipFile(TS_DIR / f"{z}_조합.zip")
    for n in zf.namelist():
        if n.endswith(".png") and "_index" not in n and "/" in n:
            combo = n.split("/")[0]
            combo_ts[combo] = z
            combo_imgs[combo].append(n)
    zf.close()
combo_tl = {}
for z in ["TL_1", "TL_3", "TL_4", "TL_5", "TL_6", "TL_7", "TL_8"]:
    zf = zipfile.ZipFile(TL_DIR / f"{z}_조합.zip")
    for n in zf.namelist():
        if n.endswith(".json"):
            combo = n.split("_json/")[0]
            combo_tl[combo] = z
    zf.close()


# 2) 조합 선정: 우리클래스 포함 + 희소 우선, 캡
def drugs(combo):
    return combo.split("-")[1:]


usable = [c for c in combo_ts if any(d in our_kc for d in drugs(c)) and c in combo_tl]
usable.sort(
    key=lambda c: (0 if any(d in rare_kc for d in drugs(c)) else 1, c)
)  # 희소 포함 먼저
sel = usable[:CAP_COMBOS]
print(f"사용가능 조합 {len(usable)} → POC 선정 {len(sel)} (희소포함 우선)", flush=True)

# 3) 추출 + 마스킹 + 변환
(OUT / "coco/images").mkdir(parents=True, exist_ok=True)
coco = {
    "images": [],
    "annotations": [],
    "categories": [{"id": c, "name": str(c)} for c in sorted(our)],
}
iid = aid = 1
zcache = {}


def zget(d, z):
    k = (d, z)
    if k not in zcache:
        zcache[k] = zipfile.ZipFile((TS_DIR if d == "ts" else TL_DIR) / f"{z}_조합.zip")
    return zcache[k]


kept_inst = Counter()
n_img = n_mask = 0
for combo in sel:
    tsz, tlz = combo_ts[combo], combo_tl[combo]
    tszf, tlzf = zget("ts", tsz), zget("tl", tlz)
    for imember in combo_imgs[combo]:
        base = Path(imember).name[:-4]  # {combo}_{params}
        # 드럭별 json 수집 → (dl_idx, bbox)
        boxes = []
        for d in drugs(combo):
            jn = f"{combo}_json/K-{d}/{base}.json"
            try:
                jd = json.load(io.BytesIO(tlzf.read(jn)))
            except KeyError:
                continue
            dl = int(
                d
            )  # 우리 category_id = K-code(subfolder). AI Hub raw dl_idx는 별개.
            anns = jd.get("annotations", [])
            if not anns or len(anns[0].get("bbox", [])) != 4:
                continue
            boxes.append((dl, [float(v) for v in anns[0]["bbox"]]))
        if not any(dl in our for dl, _ in boxes):
            continue
        # 이미지 로드
        img = cv2.imdecode(
            np.frombuffer(tszf.read(imember), np.uint8), cv2.IMREAD_COLOR
        )
        if img is None or img.shape[:2] != (H, W):
            continue
        bgmask = np.ones(
            (H, W), bool
        )  # 모든 알약(우리+비우리) 영역 제외한 실배경 median
        for _, (bx, by, bw, bh) in boxes:
            bgmask[max(0, int(by)) : int(by + bh), max(0, int(bx)) : int(bx + bw)] = (
                False
            )
        bg = (
            np.median(img[bgmask].reshape(-1, 3), 0).astype(np.uint8)
            if bgmask.any()
            else np.array([150, 150, 150], np.uint8)
        )
        anns_this = []
        for dl, (x, y, bw, bh) in boxes:
            if dl in our:
                anns_this.append((dl, [x, y, bw, bh]))
            else:  # 비-우리 알약 마스킹
                x0, y0 = max(0, int(x)), max(0, int(y))
                img[y0 : int(y + bh), x0 : int(x + bw)] = bg
                n_mask += 1
        fn = f"aihub_{base}.jpg"
        cv2.imwrite(str(OUT / "coco/images" / fn), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        coco["images"].append({"id": iid, "file_name": fn, "width": W, "height": H})
        for dl, bb in anns_this:
            coco["annotations"].append(
                {
                    "id": aid,
                    "image_id": iid,
                    "category_id": dl,
                    "bbox": bb,
                    "area": bb[2] * bb[3],
                    "iscrowd": 0,
                }
            )
            kept_inst[dl] += 1
            aid += 1
        iid += 1
        n_img += 1

json.dump(coco, open(OUT / "coco/annotations_coco.json", "w"))
print(
    f"\n>>> AI Hub POC: 이미지 {n_img}장, our-class 인스턴스 {sum(kept_inst.values())}, 클래스 {len(kept_inst)}/56, 마스킹 {n_mask}알",
    flush=True,
)
print(
    ">>> 희소보강 예시(대회<10): "
    + ", ".join(
        f"{c}:{kept_inst.get(c, 0)}" for c in sorted(our, key=lambda x: cc[x])[:8]
    ),
    flush=True,
)
print(f">>> 출력: {OUT}", flush=True)
