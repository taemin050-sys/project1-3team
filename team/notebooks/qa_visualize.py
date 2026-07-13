import json
import os
import random
from collections import Counter
import cv2
import numpy as np
from pathlib import Path

# ============================== CONFIG ==============================

current_script_dir = Path(__file__).resolve().parent

project_root = current_script_dir.parent.parent.parent

SYN_DIR = project_root / "team" / "outputs" / "synthetic_acai"

IMAGE_DIR = str(SYN_DIR / "images")
ANNOT_PATH = str(SYN_DIR / "synthetic_annotations.json")

# 시각화 결과물 역시 원격 저장소에 대량으로 올라가지 않도록 team/outputs/ 내부 공간으로 지정
QA_OUT_DIR = str(SYN_DIR / "qa_visualization")

# 레이블 박스의 투명도 (0.0: 완전 투명, 1.0: 완전 불투명) -> 그림자 관찰을 위해 반투명 추천
LABEL_ALPHA = 0.65


# ======================================================================

def main():
    os.makedirs(QA_OUT_DIR, exist_ok=True)

    if not os.path.exists(ANNOT_PATH):
        print(f"[ERROR] 어노테이션 파일이 존재하지 않습니다: {ANNOT_PATH}")
        print("build_synthetic_dataset.py를 먼저 실행해 주세요.")
        return

    with open(ANNOT_PATH, "r", encoding="utf-8") as f:
        coco_data = json.load(f)

    # 1. 카테고리 정보 로드
    categories = {c["id"]: c["name"] for c in coco_data.get("categories", [])}

    # 2. 통계 산출을 위한 데이터 그룹화
    annotations_by_img = {}
    all_category_ids = []

    for ann in coco_data.get("annotations", []):
        img_id = ann["image_id"]
        all_category_ids.append(ann["category_id"])
        if img_id not in annotations_by_img:
            annotations_by_img[img_id] = []
        annotations_by_img[img_id].append(ann)

    print(f"\n[INFO] 총 {len(coco_data['images'])}장의 이미지 시각화 및 QA 분석 가동...")

    # 고정 시드를 사용하여 실행할 때마다 클래스별로 항상 동일하고 예쁜 색상이 매핑되도록 처리
    random.seed(100)
    class_colors = {}
    for cid in categories.keys():
        class_colors[cid] = (random.randint(60, 240), random.randint(60, 240), random.randint(60, 240))

    # 3. 이미지 루프 가동
    for img_info in coco_data.get("images", []):
        img_id = img_info["id"]
        file_name = img_info["file_name"]

        img_path = os.path.join(IMAGE_DIR, file_name)
        if not os.path.exists(img_path):
            print(f"[WARN] 이미지가 디렉토리에 없습니다: {img_path}")
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        anns = annotations_by_img.get(img_id, [])

        # 각 이미지 위에 BBox 및 레이블 렌더링
        for ann in anns:
            cat_id = ann["category_id"]
            cat_name = categories.get(cat_id, f"cat_{cat_id}")
            color = class_colors.get(cat_id, (0, 255, 0))

            # BBox 좌표 추출 [x, y, w, h]
            x, y, w, h = map(int, ann["bbox"])
            x_max, y_max = x + w, y + h

            # ① 순수 알약 경계면 타이트 바운딩 박스 드로잉 (두께 2)
            cv2.rectangle(img, (x, y), (x_max, y_max), color, thickness=2)

            # ② 스마트 레이블 및 반투명 텍스트 박스 연산 파트
            label = f" {cat_name} "
            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)

            # 기본적으로 박스 바로 위쪽에 배치하되, 상단 화면 밖으로 이탈하면 박스 내부 상단으로 밀어 넣음 (방어 코드)
            if y - text_h - 8 < 0:
                ty1, ty2 = y, y + text_h + 8
                text_y = y + text_h + 3
            else:
                ty1, ty2 = y - text_h - 8, y
                text_y = y - 4

            tx1, tx2 = x, x + text_w + 4

            # 반투명 레이블 박스 효과를 위한 오버레이 레이어 블렌딩
            overlay = img.copy()
            cv2.rectangle(overlay, (tx1, ty1), (tx2, ty2), color, -1)
            cv2.addWeighted(overlay, LABEL_ALPHA, img, 1.0 - LABEL_ALPHA, 0, img)

            # 글씨 얹기 (가시성이 제일 좋은 흰색 고정 폰트 사용)
            cv2.putText(img, label, (tx1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        # 4. 시각화 완료 파일 저장
        out_path = os.path.join(QA_OUT_DIR, f"qa_{file_name}")
        cv2.imwrite(out_path, img)

    # ======================================================================
    # 5. [추가] 최종 합성 데이터셋 품질 상태 통계 리포트 브리핑
    # ======================================================================
    print("\n" + "=" * 50)
    print("      📊 SYNTHETIC DATASET QA REPORT (v2)")
    print("=" * 50)
    print(f" 🔹 총 합성 성공 이미지 수 : {len(coco_data['images'])} 장")
    print(f" 🔹 총 주입된 알약 객체 수 : {len(all_category_ids)} 개")
    print(f" 🔹 장당 평균 알약 밀집도 : {len(all_category_ids) / max(1, len(coco_data['images'])):.2f} 개/장")
    print("-" * 50)
    print(" 📂 카테고리별 알약 데이터 분포 (Class Balance Check):")

    counter = Counter(all_category_ids)
    for cid in sorted(counter.keys()):
        cname = categories.get(cid, f"Unknown_ID_{cid}")
        print(f"   - Class [{cid:02d}] {cname:<25} : {counter[cid]} 개 수집됨")
    print("=" * 50)
    print(f" 🎉 모든 시각화 검증본이 성공적으로 빌드되었습니다!")
    print(f" 👉 검증 사진 폴더: {QA_OUT_DIR}\n")


if __name__ == "__main__":
    main()