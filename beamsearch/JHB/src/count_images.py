# -*- coding: utf-8 -*-
import os
from pathlib import Path

current_script_dir = Path(__file__).resolve().parent
project_root = current_script_dir.parent.parent.parent

# ==================== 경로 설정 ====================
label_dir = str(project_root / "team" / "data" / "raw" / "acai_basic_data" / "yolo_dataset" / "final_split" / "images" / "train")

def count_images():
    if not os.path.exists(image_dir):
        print(f"❌ 경로를 찾을 수 없습니다: {image_dir}")
        return

    # 탐색할 이미지 확장자 정의 (대소문자 무관하게 처리하기 위해 소문자로 지정)
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')

    image_counter = 0
    extension_details = {}

    # 폴더 내부 탐색
    for filename in os.listdir(image_dir):
        ext = os.path.splitext(filename)[1].lower()
        if ext in valid_extensions:
            image_counter += 1
            extension_details[ext] = extension_details.get(ext, 0) + 1

    # ==================== 결과 출력 ====================
    print("📸 [이미지 데이터셋 카운트 결과]")
    print("-" * 45)
    print(f"📂 대상 폴더: {image_dir}")
    print(f"🚀 총 이미지 수 : {image_counter}장")
    print("-" * 45)

    if image_counter > 0:
        print("📊 [확장자별 세부 내역]")
        for ext, count in sorted(extension_details.items()):
            print(f"   - {ext:<7} : {count}장")
    else:
        print("⚠️ 폴더 내에 이미지 파일이 존재하지 않습니다.")
    print("-" * 45)


if __name__ == '__main__':
    count_images()