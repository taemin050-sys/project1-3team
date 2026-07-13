import json
import os
import cv2
import numpy as np
from collections import defaultdict
from pathlib import Path

# 경로 설정
current_script_dir = Path(__file__).resolve().parent

project_root = current_script_dir.parent.parent.parent

RAW_DATA_DIR = project_root / "team" / "data" / "raw" / "acai_basic_data" / "yolo_dataset" / "final_split"

json_path = str(RAW_DATA_DIR / "labels" / "train" / "train.json")  # 원하는 파일에 맞춰 수정 가능
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
                print(f"삭제 예약: {os.path.basename(target)}")
            else:
                files_to_delete.remove(target)
                print(f"삭제 취소: {os.path.basename(target)}")


def create_grid(image_paths, cat_id, img_id_to_annots, data_images_map):
    rows = int(np.ceil(len(image_paths) / GRID_SIZE))
    grid = np.zeros((rows * TILE_H, GRID_SIZE * TILE_W, 3), dtype=np.uint8)

    for i, path in enumerate(image_paths):
        img = cv2.imread(path)
        if img is not None:
            # 해당 클래스 annots 필터링
            img_id = data_images_map.get(os.path.basename(path))
            relevant_annots = [a for a in img_id_to_annots.get(img_id, []) if a['category_id'] == cat_id]

            h_orig, w_orig = img.shape[:2]
            for ann in relevant_annots:
                x, y, w, h = map(int, ann['bbox'])
                scale_x, scale_y = TILE_W / w_orig, TILE_H / h_orig
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
                cv2.putText(img, f"C:{cat_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)

            img_resized = cv2.resize(img, (TILE_W, TILE_H))
            r, c = divmod(i, GRID_SIZE)
            grid[r * TILE_H:(r + 1) * TILE_H, c * TILE_W:(c + 1) * TILE_W] = img_resized
    return grid


def visualize_by_class():
    global current_batch_paths
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 데이터 구조화
    class_to_images = defaultdict(list)
    img_id_to_annots = defaultdict(list)
    for ann in data['annotations']:
        img_id_to_annots[ann['image_id']].append(ann)
        img_info = next((img for img in data['images'] if img['id'] == ann['image_id']), None)
        if img_info:
            class_to_images[ann['category_id']].append(img_info)

    data_images_map = {os.path.basename(img['file_name'].replace("images/train/", "")): img['id'] for img in
                       data['images']}

    # 클래스 ID 정렬 후 순차적 시각화
    sorted_cat_ids = sorted(class_to_images.keys())

    for cat_id in sorted_cat_ids:
        print(f"\n>>> [검토 중] Class ID: {cat_id}")
        images_in_class = class_to_images[cat_id]
        paths = [os.path.join(image_dir, os.path.basename(img['file_name'].replace("images/train/", ""))) for img in
                 images_in_class]
        paths = [p for p in paths if os.path.exists(p)]

        for i in range(0, len(paths), GRID_SIZE ** 2):
            current_batch_paths = paths[i:i + GRID_SIZE ** 2]
            grid_img = create_grid(current_batch_paths, cat_id, img_id_to_annots, data_images_map)

            cv2.imshow(f"Class {cat_id} - [Click]:Toggle Drop, [Esc]:Next Batch, [q]:Quit", grid_img)
            cv2.setMouseCallback(f"Class {cat_id} - [Click]:Toggle Drop, [Esc]:Next Batch, [q]:Quit", mouse_callback)

            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                # 종료 시 삭제 실행
                for f in files_to_delete:
                    if os.path.exists(f): os.remove(f)
                cv2.destroyAllWindows()
                return
            elif key == 27:
                continue

    cv2.destroyAllWindows()
    for f in files_to_delete:
        if os.path.exists(f): os.remove(f)


if __name__ == "__main__":
    visualize_by_class()