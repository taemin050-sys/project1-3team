import json
import os
import cv2
import numpy as np
from pathlib import Path

current_script_dir = Path(__file__).resolve().parent

project_root = current_script_dir.parent.parent.parent

RAW_DATA_DIR = project_root / "team" / "data" / "raw" / "yolo_dataset_acai" / "final_split"

json_path = str(RAW_DATA_DIR / "labels" / "train" / "train.json")  # 파일명 규칙에 맞춰 조립
image_dir = str(RAW_DATA_DIR / "images" / "train")

# 설정
GRID_SIZE = 4
TILE_W, TILE_H = 300, 300
files_to_delete = []
current_batch_paths = []


def mouse_callback(event, x, y, flags, param):
    global files_to_delete
    if event == cv2.EVENT_LBUTTONDOWN:
        r, c = y // TILE_H, x // TILE_W
        idx = r * GRID_SIZE + c
        if idx < len(current_batch_paths):
            target = current_batch_paths[idx]
            if target not in files_to_delete:
                files_to_delete.append(target)
                print(f"삭제 예약됨: {os.path.basename(target)}")
            else:
                files_to_delete.remove(target)
                print(f"삭제 예약 취소: {os.path.basename(target)}")


def create_grid(image_paths, img_id_to_annots, data_images_map):
    grid = np.zeros((GRID_SIZE * TILE_H, GRID_SIZE * TILE_W, 3), dtype=np.uint8)
    for i, path in enumerate(image_paths):
        if i >= GRID_SIZE ** 2: break
        img = cv2.imread(path)
        if img is not None:
            # 원본 이미지에 bbox 그리기
            img_id = data_images_map.get(os.path.basename(path))
            for ann in img_id_to_annots.get(img_id, []):
                x, y, w, h = map(int, ann['bbox'])
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            img_resized = cv2.resize(img, (TILE_W, TILE_H))
            r, c = divmod(i, GRID_SIZE)
            grid[r * TILE_H:(r + 1) * TILE_H, c * TILE_W:(c + 1) * TILE_W] = img_resized
    return grid


def run_viewer():
    global current_batch_paths
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    annotations = data.get('annotations', [])
    img_id_to_annots = {}
    for ann in annotations:
        img_id_to_annots.setdefault(ann['image_id'], []).append(ann)

    data_images_map = {os.path.basename(img['file_name'].replace("images/train/", "")): img['id'] for img in
                       data['images']}
    image_paths = [os.path.join(image_dir, os.path.basename(img['file_name'].replace("images/train/", ""))) for img in
                   data['images']]

    for i in range(0, len(image_paths), GRID_SIZE ** 2):
        current_batch_paths = image_paths[i:i + GRID_SIZE ** 2]
        grid_img = create_grid(current_batch_paths, img_id_to_annots, data_images_map)

        cv2.imshow("16-Grid Viewer - [Click]: Toggle Delete, [Esc]: Next, [q]: Finish", grid_img)
        cv2.setMouseCallback("16-Grid Viewer - [Click]: Toggle Delete, [Esc]: Next, [q]: Finish", mouse_callback)

        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
        elif key == 27:
            continue

    print(f"\n총 {len(files_to_delete)}개의 파일을 삭제합니다...")
    for f in files_to_delete:
        if os.path.exists(f): os.remove(f)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_viewer()