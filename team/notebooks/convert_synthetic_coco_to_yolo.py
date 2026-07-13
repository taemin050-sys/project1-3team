import json
import os
import shutil
from pathlib import Path

current_script_dir = Path(__file__).resolve().parent

project_root = current_script_dir.parent.parent.parent

SYN_DIR = project_root / "team" / "outputs" / "synthetic_acai"
JSON_PATH = str(SYN_DIR / "synthetic_annotations.json")

OUTPUT_DIR = str(SYN_DIR / "labels")
# ========================================================


def convert_synthetic_labels():
    # 1. 목적지 폴더 무결성 리셋 (오염 방지)
    if os.path.exists(OUTPUT_DIR):
        print(f"🧹 기존 데이터 혼선을 막기 위해 폴더 초기화 중: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 2. JSON 파일 로드
    if not os.path.exists(JSON_PATH):
        print(f"❌ 합성 정답 JSON 파일을 찾을 수 없습니다. 경로를 확인하세요: {JSON_PATH}")
        return

    print("⏳ 대용량 합성 정답 JSON 파일을 읽어오는 중입니다...")
    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)

    # 3. 이미지 메타데이터 맵 구축 (id -> {file_name, width, height})
    image_map = {}
    for img in coco_data['images']:
        image_map[img['id']] = {
            'file_name': img['file_name'],
            'width': img['width'],
            'height': img['height']
        }

    # 4. 이미지 ID별로 어노테이션(알약 박스들) 그룹화
    annotations_by_image = {img_id: [] for img_id in image_map.keys()}
    for annot in coco_data['annotations']:
        img_id = annot['image_id']
        if img_id in annotations_by_image:
            annotations_by_image[img_id].append(annot)

    print("🚀 YOLO 포맷 .txt 라벨 파일 변환 및 분할을 시작합니다...")
    success_count = 0
    total_objects = 0

    # 5. 역정규화 및 정밀 비율 연산 기동
    for img_id, annots in annotations_by_image.items():
        img_info = image_map[img_id]
        img_w = img_info['width']
        img_h = img_info['height']

        # 확장자 분리하여 동일 이름의 .txt 파일명 매칭 (syn_000001.png -> syn_000001.txt)
        base_name = os.path.splitext(img_info['file_name'])[0]
        txt_filename = f"{base_name}.txt"
        txt_path = os.path.join(OUTPUT_DIR, txt_filename)

        yolo_lines = []
        for annot in annots:
            class_id = annot['category_id']  # 이미 합성 단계에서 0~92 정제 완료됨

            # COCO bbox: [xmin, ymin, width, height]
            bbox = annot['bbox']
            xmin, ymin, w, h = bbox[0], bbox[1], bbox[2], bbox[3]

            # YOLO 포맷 정규화 공식 가동 (중심X, 중심Y, 너비, 높이를 0~1 비율로 변환)
            x_center = (xmin + w / 2.0) / img_w
            y_center = (ymin + h / 2.0) / img_h
            norm_w = w / img_w
            norm_h = h / img_h

            # 소수점 6자리까지 타이트하게 문자열 포맷팅
            yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")
            total_objects += 1

        # 바운딩 박스가 존재하는 안전 데이터만 텍스트 파일로 저장
        if yolo_lines:
            with open(txt_path, 'w', encoding='utf-8') as txt_f:
                txt_f.write("\n".join(yolo_lines))
            success_count += 1

    print("\n" + "=" * 50)
    print("🎉 [합성 데이터셋 라벨 변환 완료]")
    print(f"📂 정상 저장된 .txt 라벨 파일 수 : {success_count}개")
    print(f"📊 텍스트 파일에 기입된 총 알약 수: {total_objects}개")
    print(f"💾 보관 경로: {OUTPUT_DIR}")
    print("=" * 50)


if __name__ == '__main__':
    convert_synthetic_labels()