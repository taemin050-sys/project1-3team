"""Codex 핸드오프 번들 생성: 56클래스 목록 + 스펙 + (1)배경패치 + 도메인프로파일 + 참고이미지 → zip."""

import json
import csv
import random
import shutil
import zipfile
from pathlib import Path
from collections import Counter
import numpy as np
import cv2

BASE = Path("/Users/macbook/dev/learning/codeit/01_Proj_HealthEat_cla")
LHK = BASE / "project1-3team/beamsearch/LHK"
YD, SSOT = LHK / "data/yolo", LHK / "data/processed"
OUT = BASE / "codex_handoff"
W, H = 976, 1280
if OUT.exists():
    shutil.rmtree(OUT)
(OUT / "backgrounds").mkdir(parents=True)
(OUT / "refs").mkdir()

cm = json.load(open(SSOT / "class_map.json"))
dm = json.load(open(SSOT / "drug_master.json"))
c2m = cm["category_id_to_model_index"]  # str(dl_idx) -> model_index

# ---------- 1) class_map_56 (SSOT for Codex) ----------
rows = []
for cid_s, mi in sorted(c2m.items(), key=lambda x: x[1]):
    cid = int(cid_s)
    d = dm.get(cid_s, {})
    rows.append(
        {
            "class_index": mi,
            "category_id": cid,
            "K_code": f"K-{cid:06d}",
            "product_name": d.get("product_name"),
            "ingredient": d.get("ingredient"),
            "di_class_no": d.get("class_no"),
            "otc": d.get("otc_code"),
            "shape": d.get("shape"),
            "color": d.get("color"),
            "imprint_front": d.get("imprint_front"),
            "item_seq": d.get("item_seq"),
        }
    )
with open(OUT / "class_map_56.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
json.dump(
    {"num_classes": 56, "classes": rows},
    open(OUT / "class_map_56.json", "w"),
    ensure_ascii=False,
    indent=1,
)

# ---------- 2) (1) 이미지+bbox 로드 (232) ----------
imgs = []
for sp in ("train", "val"):
    for p in sorted((YD / "images" / sp).glob("*.png")):
        boxes = []
        for ln in (YD / "labels" / sp / (p.stem + ".txt")).read_text().splitlines():
            if not ln.strip():
                continue
            m, cx, cy, nw, nh = ln.split()
            cx, cy, nw, nh = map(float, (cx, cy, nw, nh))
            boxes.append((int(m), (cx - nw / 2) * W, (cy - nh / 2) * H, nw * W, nh * H))
        imgs.append((p, boxes))

rng = random.Random(42)

# ---------- 3) (1) 배경 RGB + 깨끗한 패치 16장 ----------
bg_means = []
saved = 0
for p, boxes in rng.sample(imgs, len(imgs)):
    im = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
    mask = np.ones((H, W), bool)
    for _, x, y, w_, h_ in boxes:
        x, y, w_, h_ = int(x), int(y), int(w_), int(h_)
        mask[max(0, y - 20) : y + h_ + 20, max(0, x - 20) : x + w_ + 20] = False
    bg_means.append(im[mask].reshape(-1, 3).mean(0))
    if saved < 16:
        for _ in range(6):
            py, px = rng.randint(0, H // 3), rng.randint(0, W - 256)
            patch = im[py : py + 256, px : px + 256]
            if patch.shape[:2] != (256, 256):
                continue
            pm = patch.reshape(-1, 3).mean(0)
            if (
                pm[2] > pm[0] and 70 < pm.mean() < 175
            ):  # 블루그레이 필터(마커·비네팅 배제)
                cv2.imwrite(
                    str(OUT / "backgrounds" / f"bg_{saved:02d}.png"),
                    cv2.cvtColor(patch, cv2.COLOR_RGB2BGR),
                )
                saved += 1
                break
bg_rgb = np.array(bg_means).mean(0).round().astype(int)

# ---------- 4) 도메인 프로파일 ----------
areas, sides, nobj = [], [], []
for p, boxes in imgs:
    nobj.append(len(boxes))
    for _, x, y, w_, h_ in boxes:
        areas.append(w_ * h_ / (W * H))
        sides.append((w_ * h_) ** 0.5)


def pct(a):
    return {
        str(q): round(float(np.quantile(a, q)), 3)
        for q in (0.05, 0.25, 0.5, 0.75, 0.95)
    }


profile = {
    "resolution_WxH": [W, H],
    "background_rgb_target_RGB": bg_rgb.tolist(),
    "background_note": "실제 (1) 배경은 블루그레이(B>R). backgrounds/*.png 를 샘플·타일 권장.",
    "bbox_rel_area_ratio_pct": pct(areas),
    "bbox_side_px_pct": pct(sides),
    "objects_per_image": {str(k): v for k, v in sorted(Counter(nobj).items())},
    "camera_la_seen": [70, 75, 90],
    "camera_lo_seen": [0],
    "orientation": "앞면 위주",
    "lighting": "주백색(neutral)",
}
json.dump(profile, open(OUT / "domain_profile.json", "w"), ensure_ascii=False, indent=1)

# ---------- 5) 참고 실제 조합 이미지 4장(+bbox) ----------
for i, (p, boxes) in enumerate(rng.sample(imgs, 4)):
    im = cv2.imread(str(p))
    cv2.imwrite(str(OUT / "refs" / f"real_combo_{i}.jpg"), im)
    v = im.copy()
    for _, x, y, w_, h_ in boxes:
        cv2.rectangle(v, (int(x), int(y)), (int(x + w_), int(y + h_)), (0, 255, 0), 3)
    cv2.imwrite(str(OUT / "refs" / f"real_combo_{i}_bbox.jpg"), v)

# ---------- 6) SPEC.md ----------
spec = f"""# Codex 데이터 생성 스펙 — HealthEat 캐글 대회용 조합 증강 (56 클래스)

## 목표
AI Hub 경구약제 **단일**을 **우리 대회 56 클래스로 필터**해, **(1) 대회 도메인에 맞는 조합 이미지**로 합성. **train 전용**.

## ⛔ 절대 규칙 (오해·오류 금지)
1. **클래스 = `class_map_56.csv`의 56개만.** `K_code` 목록 밖 제품은 **절대 사용 금지**. (기존 v2는 임의 79종이라 우리 대회와 1종만 겹쳐 무효였음.)
2. **라벨 category_id = 해당 제품 dl_idx**(csv의 `category_id`, 예 `1900`). **`category_id=1` 절대 금지.** 모델용 `class_index`(0–55)와 `product_id`(=K_code)도 csv대로.
3. **배경 = (1) 도메인 블루그레이.** `backgrounds/*.png`(실제 (1) 배경 패치)를 샘플·타일하거나 target RGB≈`{bg_rgb.tolist()}`(±15). **어둡거나 다채로운 배경(자주·검정·올리브·탄 등) 금지** — 그건 서비스용 v1이었고 대회 도메인과 불일치.
4. **금지 데이터:** `TL_2_조합`·`TS_2_조합` 절대 미사용. **단일(single)만** 사용.
5. **train 전용.** val/test에 절대 미투입(우리 val=실제 (1) 조합).

## 소스
- AI Hub 단일 5000종 중 **`K_code ∈ class_map_56`** 인 것만 필터. 클래스당 다양한 각도/조명 표본.

## 컷아웃 (기존 SAM2 유지 — 품질 우수)
- **SAM2(sam2.1_hiera_large, MPS)** 그대로. 마스크 **1–2px 침식**(fringe 제거). 각 알약 **WB 정규화**(주백색 중성 톤).

## 합성 (도메인 매칭)
- 캔버스 **{W}×{H}**. 배경 = `backgrounds/` 패치 랜덤.
- **2–4알**, **비겹침 그리드+지터**(테스트 저오클루전). **앞면 우선**.
- 알약 크기: `domain_profile.json` 분포 준수(상대면적 중앙값 ~{pct(areas)["0.5"]}, 한 변 중앙값 ~{pct(sides)["0.5"]:.0f}px).
- 조명·그림자 약하게(주백색 단일 조건).

## 클래스 균형·규모
- **희소 클래스 우선**(대회 train은 48/56이 <20샘플). **클래스당 최소 ~50표본**까지 보강.
- 전체 synth **≤ real(232)의 2–3배**. 특정 약 편향 재현 금지.

## 라벨 출력
- WebDataset(jpg+json) 또는 COCO. json은 v2 스키마 유지 가능하나 **`category_id`를 반드시 dl_idx로**.
- 각 annotation: `{{id, bbox:[x,y,w,h]px, area, category_id:<dl_idx>, class_index:<0-55>, product_id:"K-00XXXX"}}`.

## 자동 검수(생성 후)
- 모든 `category_id ∈ 56`(csv) / `class_index ∈ 0..55`.
- bbox 유효(길이4·양수·이미지 내).
- **배경 RGB가 target 근방**(어둡/다채 아님) 샘플 확인.

## 동봉물
- `class_map_56.csv|json` — 클래스 SSOT(class_index↔category_id(dl_idx)↔K_code↔약정보).
- `backgrounds/*.png` — 실제 (1) 블루그레이 배경 패치 {saved}장.
- `domain_profile.json` — 정량 목표(배경RGB·bbox분포·객체수·해상도·각도).
- `refs/real_combo_*.jpg` — 실제 (1) 대회 이미지(목표 도메인), `_bbox`는 라벨 예시.

> 생성물 오면 우리가 **fold0 하니스(mAP@[0.75:0.95])** 로 baseline 대비 애블레이션(다채배경 v2 vs 이 블루그레이본)으로 효과를 숫자 판정.
"""
(OUT / "SPEC.md").write_text(spec, encoding="utf-8")

# ---------- 7) zip ----------
zp = BASE / "codex_handoff.zip"
if zp.exists():
    zp.unlink()
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
    for f in OUT.rglob("*"):
        if f.is_file() and f.name != ".DS_Store":
            z.write(f, f.relative_to(OUT.parent))
print("배경 RGB target:", bg_rgb.tolist(), "| 배경패치:", saved, "장")
print("객체수 분포:", profile["objects_per_image"])
print("번들:", zp, f"({zp.stat().st_size / 1e6:.1f} MB)")
print("포함:", sorted(p.name for p in OUT.iterdir()))
