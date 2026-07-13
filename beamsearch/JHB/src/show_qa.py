# -*- coding: utf-8 -*-
import cv2
import os
from pathlib import Path

# ============================== CONFIG ==============================
current_script_dir = Path(__file__).resolve().parent

project_root = current_script_dir.parent.parent.parent

QA_OUT_DIR = str(project_root / "team" / "outputs" / "synthetic_acai" / "qa_visualization")
# ======================================================================

def show_images():
    # 폴더 내 파일 리스트 정렬 (qa_syn_000001.png 순서대로)
    if not os.path.exists(QA_OUT_DIR):
        print(f"[ERROR] 경로가 존재하지 않습니다. 먼저 QA 스크립트를 가동해 주세요: {QA_OUT_DIR}")
        return

    files = sorted([f for f in os.listdir(QA_OUT_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    if not files:
        print(f"[ERROR] 시각화된 이미지가 폴더에 없습니다: {QA_OUT_DIR}")
        return

    print(f"[INFO] 총 {len(files)}장의 이미지를 로드합니다.")
    print(" - 스페이스바: 다음 사진")
    print(" - ESC: 종료")

    for i, file_name in enumerate(files):
        img_path = os.path.join(QA_OUT_DIR, file_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        # 화면에 너무 큰 이미지가 뜨지 않도록 적절히 리사이즈 (가로 1000 고정)
        h, w = img.shape[:2]
        display_w = 1000
        display_h = int(h * (display_w / w))
        img_display = cv2.resize(img, (display_w, display_h))

        cv2.imshow("QA Viewer (Space: Next, ESC: Exit)", img_display)

        key = cv2.waitKey(0)  # 키 입력 대기 (0은 무한 대기)

        if key == 27:  # ESC 키
            print("[INFO] 뷰어를 종료합니다.")
            break
        elif key == 32:  # 스페이스바
            continue

    cv2.destroyAllWindows()


if __name__ == "__main__":
    show_images()