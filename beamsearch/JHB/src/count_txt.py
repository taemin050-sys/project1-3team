# -*- coding: utf-8 -*-
import os
from pathlib import Path

current_script_dir = Path(__file__).resolve().parent
project_root = current_script_dir.parent.parent.parent

# ==================== 경로 설정 ====================
label_dir = str(project_root / "team" / "data" / "raw" / "acai_basic_data" / "yolo_dataset" / "final_split" / "labels" / "train")


def count_labels():
    if not os.path.exists(label_dir):
        print(f"❌ 경로를 찾을 수 없습니다: {label_dir}")
        return

    txt_counter = 0

    # 폴더 내부 탐색하여 .txt 파일만 필터링
    for filename in os.listdir(label_dir):
        if filename.lower().endswith('.txt'):
            txt_counter += 1

    # ==================== 결과 출력 ====================
    print("📝 [라벨 데이터셋 카운트 결과]")
    print("-" * 45)
    print(f"📂 대상 폴더: {label_dir}")
    print(f"🚀 총 라벨(.txt) 파일 수 : {txt_counter}장")
    print("-" * 45)

    if txt_counter == 0:
        print("⚠️ 폴더 내에 .txt 파일이 존재하지 않습니다.")
        print("-" * 45)


if __name__ == '__main__':
    count_labels()