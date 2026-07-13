import json
import os
import random
import shutil
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

current_script_dir = Path(__file__).resolve().parent
project_root = current_script_dir.parent.parent.parent
jhb_root_dir = current_script_dir.parent
config_dir = jhb_root_dir / "configs"

bg_images = [
    str(config_dir / "empty_bg_1.png"),
    str(config_dir / "empty_bg_2.png"),
    str(config_dir / "empty_bg_3.png"),
    str(config_dir / "empty_bg_4.png"),
    str(config_dir / "empty_bg_5.png"),
    str(config_dir / "empty_bg_6.png"),
    str(config_dir / "empty_bg_7.png"),
]

# ============================== CONFIG ==============================
IMAGE_DIR = str(project_root / "team" / "data" / "raw" / "yolo_dataset_acai" / "final_split" / "images" / "train")
LABEL_DIR = str(project_root / "team" / "data" / "raw" / "yolo_dataset_acai" / "final_split" / "labels" / "train")

# 출력 디렉토리 파트
OUT_DIR = project_root / "team" / "outputs" / "synthetic_acai"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 하위 자산 폴더 경로 지정
PILL_BANK_DIR = OUT_DIR / "pill_bank"
SYN_IMAGE_DIR = OUT_DIR / "images"
SYN_ANNOT_PATH = OUT_DIR / "synthetic_annotations.json"

PILL_BANK_DIR.mkdir(parents=True, exist_ok=True)
SYN_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

# 문자열 변환 변수들 배치
OUT_DIR_STR = str(OUT_DIR)
PILL_BANK_DIR_STR = str(PILL_BANK_DIR)
SYN_IMAGE_DIR_STR = str(SYN_IMAGE_DIR)
SYN_ANNOT_PATH_STR = str(SYN_ANNOT_PATH)

CROP_PADDING = 30
FEATHER_PX = 4
USE_CONVEX_HULL = True

MAX_SHADOW_DIST = 60
MIN_SHADOW_DIST = 8
MAX_SHADOW_BLUR = 141
MIN_SHADOW_BLUR = 25
SHADOW_OPACITY_RANGE = (0.22, 0.48)

MIN_GAP_PX = 10

# 클래스당 합성 알약 목표 객체 수 (상한 300개 설정)
TARGET_INSTANCES_PER_CAT = 300

CLASS_NAMES = {
    0: "마그밀정(수산화마그네슘)",
    1: "게보린정 300mg/PTP",
    2: "알마겔정(알마게이트)(수출명:유한가스트라겔정)",
    3: "보령부스파정 5mg",
    4: "뮤테란캡슐 100mg",
    5: "일양하이트린정 2mg",
    6: "기넥신에프정(은행엽엑스)(수출용)",
    7: "무코스타정(레바미피드)(비매품)",
    8: "동아오팔몬정(리마프로스트알파-시클로덱스트린포접화합물)",
    9: "알드린정",
    10: "뉴로메드정(옥시라세탐)",
    11: "타이레놀정500mg",
    12: "에어탈정(아세클로페낙)",
    13: "비유피-4정 20mg",
    14: "엘도스캡슐(에르도스테인)(수출용)",
    15: "삼남건조수산화알루미늄겔정",
    16: "프로스카정",
    17: "타이레놀이알서방정(아세트아미노펜)(수출용)",
    18: "삐콤씨에프정 618.6mg/병",
    19: "조인스정 200mg",
    20: "쎄로켈정 100mg",
    21: "넥시움정 40mg",
    22: "아스피린프로텍트정 100mg",
    23: "리렉스펜정 300mg/PTP",
    24: "아빌리파이정 10mg",
    25: "자이프렉사정 2.5mg",
    26: "다보타민큐정 10mg/병",
    27: "엘스테인캡슐(에르도스테인)",
    28: "써스펜8시간이알서방정 650mg",
    29: "에빅사정(메만틴염산염)(비매품)",
    30: "한미탐스캡슐 0.2mg",
    31: "아보다트연질캡슐 0.5mg",
    32: "리피토정 20mg",
    33: "크레스토정 20mg",
    34: "가바토파정 100mg",
    35: "동아가바펜틴정 800mg",
    36: "오마코연질캡슐(오메가-3-산에틸에스테르90)",
    37: "란스톤엘에프디티정 30mg",
    38: "리리카캡슐 150mg",
    39: "종근당글리아티린연질캡슐(콜린알포세레이트)",
    40: "콜리네이트연질캡슐 400mg",
    41: "트루비타정 60mg/병",
    42: "스토가정 10mg",
    43: "노바스크정 5mg",
    44: "마도파정",
    45: "플라빅스정 75mg",
    46: "자트랄엑스엘정 10mg",
    47: "베시케어정 10mg",
    48: "엑스포지정 5/160mg",
    49: "펠루비정(펠루비프로펜)",
    50: "아토르바정 10mg",
    51: "라비에트정 20mg",
    52: "리피로우정 20mg",
    53: "자누비아정 50mg",
    54: "맥시부펜이알정 300mg",
    55: "메가파워정 90mg/병",
    56: "쿠에타핀정 25mg",
    57: "비타비백정 100mg/병",
    58: "토비애즈서방정 4mg",
    59: "놀텍정 10mg",
    60: "자누메트정 50/850mg",
    61: "큐시드정 31.5mg/PTP",
    62: "아모잘탄정 5/100mg",
    63: "세비카정 10/40mg",
    64: "트윈스타정 40/5mg",
    65: "카나브정 60mg",
    66: "울트라셋이알서방정",
    67: "졸로푸트정 100mg",
    68: "플리바스정 50mg",
    69: "트라젠타정(리나글립틴)",
    70: "비모보정 500/20mg",
    71: "레일라정",
    72: "리바로정 4mg",
    73: "렉사프로정 15mg",
    74: "트라젠타듀오정 2.5/850mg",
    75: "낙소졸정 500/20mg",
    76: "아질렉트정(라사길린메실산염)",
    77: "자누메트엑스알서방정 100/1000mg",
    78: "글리아타민연질캡슐",
    79: "신바로정",
    80: "트루패스정 4mg",
    81: "에스원엠프정 20mg",
    82: "브린텔릭스정 20mg",
    83: "글리틴정(콜린알포세레이트)",
    84: "제미메트서방정 50/1000mg",
    85: "아토젯정 10/40mg",
    86: "로수젯정10/5밀리그램",
    87: "알바스테인캡슐(에르도스테인)",
    88: "로수바미브정 10/20mg",
    89: "뮤코원캡슐(에르도스테인)",
    90: "카발린캡슐 25mg",
    91: "케이캡정 50mg",
    92: "엘스테인정(에르도스테인)"
}

# 데이터 개수가 많은 상위 다수 클래스 목록 전체 고정 (300개 이상 확보된 클래스 정제 목록)
EXCLUDE_CLASSES = [34, 35, 22, 8, 9, 26, 41, 2]
PILLS_PER_IMAGE_RANGE = (3, 4)
CANVAS_SIZE = (976, 1280)
MAX_PLACEMENT_TRIES = 500
SANITY_CHECK_ONLY = False
# ======================================================================

_EXCLUDE_SET_STR = {str(x) for x in EXCLUDE_CLASSES}


def is_excluded_category(cat_id) -> bool:
    """합성 대상에서 제외할 대형(다수) 클래스인지 판별"""
    return str(cat_id) in _EXCLUDE_SET_STR or int(cat_id) in EXCLUDE_CLASSES


def load_user_backgrounds(bg_paths: list):
    """실제 존재하는 배경 이미지 경로만 필터링하여 로드"""
    valid = [p for p in bg_paths if os.path.exists(p)]
    return valid


def apply_directional_lighting(rgba: np.ndarray, light_angle_deg: float, max_shading: float = 0.22):
    """
        [광학 증강] 알약 객체에 무작위 각도의 사선 광원(음영 그라데이션)을 부여하여
        입체감을 살리고 합성 티가 나는 것을 억제함
    """
    h, w = rgba.shape[:2]
    bgr = rgba[:, :, :3].astype(np.float32)
    alpha = rgba[:, :, 3]

    # 이미지 중심 기준 격자 좌표 생성
    y, x = np.indices((h, w))
    center_y, center_x = h / 2.0, w / 2.0
    y = y - center_y
    x = x - center_x

    angle_rad = np.radians(light_angle_deg)
    proj = x * np.cos(angle_rad) + y * np.sin(angle_rad)

    # 지정된 광원 각도로 정사영(Projection)하여 선형 그라데이션 맵 계산
    max_val = np.max(np.abs(proj)) if np.max(np.abs(proj)) > 0 else 1.0
    norm_proj = proj / max_val

    # 광원 방향은 밝게, 반대 방향은 어둡게
    lighting_gradient = 1.0 + (norm_proj * max_shading)
    lighting_gradient_3ch = np.dstack([lighting_gradient] * 3)

    shaded_bgr = bgr * lighting_gradient_3ch
    shaded_bgr = np.clip(shaded_bgr, 0, 255).astype(np.uint8)
    return np.dstack([shaded_bgr, alpha])


def collect_annotations_from_yolo_txt():
    """
        [데이터 로드] 원본 학습 세트의 YOLO 정답지(.txt)를 파싱하여
        실제 픽셀 크기(Absolute Coordinates)의 Bounding Box 정보로 변환 및 수집
    """
    merged = {}
    total_parsed_objects = 0
    valid_extensions = ('.png', '.jpg', '.jpeg')
    raw_files = os.listdir(IMAGE_DIR)
    image_names = [f for f in raw_files if os.path.splitext(f)[1].lower() in valid_extensions]

    for img_name in image_names:
        base_name = os.path.splitext(img_name)[0]
        txt_name = f"{base_name}.txt"
        txt_path = os.path.join(LABEL_DIR, txt_name)
        if not os.path.exists(txt_path):
            continue

        merged[img_name] = []
        with open(txt_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            class_id = int(parts[0])

            img_full_path = os.path.join(IMAGE_DIR, img_name)
            temp_img = cv2.imread(img_full_path)
            if temp_img is None:
                continue
            img_h, img_w = temp_img.shape[:2]

            x_c, y_c, w_ratio, h_ratio = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

            w = w_ratio * img_w
            h = h_ratio * img_h
            xmin = (x_c * img_w) - (w / 2.0)
            ymin = (y_c * img_h) - (h / 2.0)

            merged[img_name].append({
                "bbox": [xmin, ymin, w, h],
                "category_id": class_id,
                "category_name": CLASS_NAMES.get(class_id, f"Class_{class_id}")
            })
            total_parsed_objects += 1

    print(f"🔍 [YOLO TXT 스캔 결과] 고유 알약 정답 {total_parsed_objects}개 확보 완료.")
    return merged


def extract_mask_sam(sam_model, patch_bgr: np.ndarray, orig_box_local) -> np.ndarray:
    """
        [마스크 추출] MobileSAM을 활용하여 알약 영역의 정밀 세그멘테이션 마스크(누끼)를 땁니다
        추출 후 외곽선(Contour) 기반 Convex Hull을 적용해 노이즈 없는 매끄러운 마스크 완성!
    """
    H, W = patch_bgr.shape[:2]
    ox, oy, ow, oh = orig_box_local
    x1, y1 = max(0, ox - 1), max(0, oy - 1)
    x2, y2 = min(W, ox + ow + 1), min(H, oy + oh + 1)
    sam_bbox = [x1, y1, x2, y2]
    results = sam_model(patch_bgr, bboxes=[sam_bbox], verbose=False)
    if len(results) == 0 or results[0].masks is None:
        fallback = np.zeros((H, W), dtype=np.uint8)
        fallback[oy:oy + oh, ox:ox + ow] = 255
        return fallback
    sam_mask = results[0].masks.data[0].cpu().numpy()
    binary_mask = (sam_mask * 255).astype(np.uint8)
    if binary_mask.shape[:2] != (H, W):
        binary_mask = cv2.resize(binary_mask, (W, H), interpolation=cv2.INTER_NEAREST)

    # 알약 외곽을 매끄럽게 보정하기 위한 Convex Hull(볼록 껍질) 처리; 검출한 외곽선 기준 오목한 부분들을 볼록하게 채워 넣는 작업입니다
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        clean_mask = np.zeros_like(binary_mask)
        if USE_CONVEX_HULL:
            hull = cv2.convexHull(largest_contour)
            cv2.drawContours(clean_mask, [hull], -1, 255, thickness=cv2.FILLED)
        else:
            cv2.drawContours(clean_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
        binary_mask = clean_mask
    return binary_mask


def tight_bbox_from_mask(mask: np.ndarray):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    return int(x_min), int(y_min), int(x_max - x_min + 1), int(y_max - y_min + 1)


def build_pill_bank_with_sam(merged_annotations: dict, image_dir: str, bank_dir: str):
    """
        [자산 구축] 전체 이미지에서 SAM으로 누끼를 전수 추출하여
        배경이 투명한 RGBA(PNG) 형태의 알약 자산 창고(Pill Bank)를 빌드
    """
    print("[INFO] 🚀 최초 마스터 마스크 추출을 위한 MobileSAM 로딩 가동...")
    from ultralytics import SAM
    sam_model = SAM('mobile_sam.pt')

    os.makedirs(bank_dir, exist_ok=True)
    records = []
    idx = 0

    for file_name, anns in tqdm(merged_annotations.items(), desc="Pill 누끼 진행 중"):
        img_path = os.path.join(image_dir, file_name)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            continue
        H, W = img.shape[:2]

        for ann in anns:
            idx += 1
            x, y, w, h = ann["bbox"]

            # 크롭 시 여백(CROP_PADDING)을 주어 그림자나 경계면이 잘리지 않도록 함
            x0, y0 = max(0, int(x - CROP_PADDING)), max(0, int(y - CROP_PADDING))
            x1, y1 = min(W, int(x + w + CROP_PADDING)), min(H, int(y + h + CROP_PADDING))
            patch = img[y0:y1, x0:x1].copy()
            if patch.size == 0:
                continue

            orig_box_local = (int(x - x0), int(y - y0), int(w), int(h))
            mask = extract_mask_sam(sam_model, patch, orig_box_local)
            tb = tight_bbox_from_mask(mask)
            if tb is None:
                continue

            tx, ty, tw, th = tb

            # 마스크가 비정상적으로 작거나 크면 오검출로 판단하여 필터링
            orig_area = w * h
            mask_area = tw * th
            if mask_area < 0.2 * orig_area or mask_area > 2.0 * orig_area:
                continue

            # 여백을 제거하고 알약 크기에 딱 맞춘 투명 채널(RGBA) 이미지 저장
            pill_rgb = patch[ty:ty + th, tx:tx + tw]
            pill_mask = mask[ty:ty + th, tx:tx + tw]
            rgba = cv2.cvtColor(pill_rgb, cv2.COLOR_BGR2BGRA)
            rgba[:, :, 3] = pill_mask
            out_name = f"pill_{idx:06d}_cat{ann['category_id']}.png"
            out_path = os.path.join(bank_dir, out_name)
            cv2.imwrite(out_path, rgba)

            records.append({
                "rgba_path": out_path,
                "category_id": ann["category_id"],
                "category_name": ann["category_name"],
            })
    return records


def rotate_rgba(rgba: np.ndarray, angle: float) -> np.ndarray:
    """[기하학 증강] 알약을 무작위 각도로 회전시키며, 회전 시 외곽선 잘림을 방지하기 위해 대각선 크기의 캔버스를 임시 사용"""
    h, w = rgba.shape[:2]
    diag = int(np.ceil(np.sqrt(h ** 2 + w ** 2)))
    canvas = np.zeros((diag, diag, 4), dtype=np.uint8)
    y0, x0 = (diag - h) // 2, (diag - w) // 2
    canvas[y0:y0 + h, x0:x0 + w] = rgba
    M = cv2.getRotationMatrix2D((diag / 2, diag / 2), angle, 1.0)
    rotated = cv2.warpAffine(canvas, M, (diag, diag), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    alpha = rotated[:, :, 3]
    _, alpha_thresh = cv2.threshold(alpha, 200, 255, cv2.THRESH_BINARY)
    rotated[:, :, 3] = alpha_thresh
    tb = tight_bbox_from_mask(alpha_thresh)
    if tb is None:
        return rgba
    tx, ty, tw, th = tb
    return rotated[ty:ty + th, tx:tx + tw]


def boxes_overlap_with_shadow_safety(box_a, box_b, shadow_safe_margin: int) -> bool:
    """[위치 검증] 알약 배치 시 다른 알약 및 생성될 그림자 반경(shadow_safe_margin)과 겹치지 않는지 체크"""
    xa, ya, wa, ha = box_a
    xb, yb, wb, hb = box_b
    ax0, ay0 = xa - shadow_safe_margin, ya - shadow_safe_margin
    ax1, ay1 = xa + wa + shadow_safe_margin, ya + ha + shadow_safe_margin
    return not (ax1 <= xb or ax0 >= xb + wb or ay1 <= yb or ay0 >= yb + hb)


def paste_rgba_with_smart_shadow(canvas_bgr: np.ndarray, rgba: np.ndarray, x: int, y: int, custom_light_angle=None):
    """
        [고도화 블렌딩] 알약 객체의 조명 각도 및 명도 통계를 분석하여
        자연스러운 그림자(방향성, 흐림 효과, 투명도 조절)를 먼저 캔버스에 렌더링한 후
        알약을 페더링(깃털 효과) 처리하여 합성
    """
    h, w = rgba.shape[:2]
    H, W = canvas_bgr.shape[:2]

    rgb_pill = rgba[:, :, :3]
    alpha_raw = rgba[:, :, 3]
    gray_pill = cv2.cvtColor(rgb_pill, cv2.COLOR_BGR2GRAY)
    mask_indices = np.where(alpha_raw > 0)

    offset_x, offset_y = 0, 8
    blur_size = 35
    opacity = 0.35

    # 1. 스마트 그림자 파라미터 계산 (알약 내부 명암 분포 혹은 지정된 광원 각도 추적)
    if len(mask_indices[0]) > 10:
        tilt_ratio = np.clip((np.std(gray_pill[mask_indices]) - 5.0) / 40.0, 0.0, 1.0)
        shadow_dist = MIN_SHADOW_DIST + int((MAX_SHADOW_DIST - MIN_SHADOW_DIST) * tilt_ratio)

        if custom_light_angle is not None:
            # 광원 반대 방향으로 그림자 오프셋 설정
            shadow_angle_rad = np.radians(custom_light_angle + 180)
            offset_x = int(round(np.cos(shadow_angle_rad) * shadow_dist))
            offset_y = int(round(np.sin(shadow_angle_rad) * shadow_dist))
        else:
            # 밝기 무게중심을 이용해 어두운 쪽으로 그림자 방향 유도
            center_y, center_x = np.mean(mask_indices[0]), np.mean(mask_indices[1])
            darkness_weights = 255.0 - gray_pill[mask_indices].astype(np.float32)
            sum_weights = np.sum(darkness_weights) + 1e-5
            shadow_center_y = np.sum(mask_indices[0] * darkness_weights) / sum_weights
            shadow_center_x = np.sum(mask_indices[1] * darkness_weights) / sum_weights
            vec_x, vec_y = shadow_center_x - center_x, shadow_center_y - center_y
            vec_len = np.sqrt(vec_x ** 2 + vec_y ** 2) + 1e-5
            offset_x = int(round((vec_x / vec_len) * shadow_dist))
            offset_y = int(round((vec_y / vec_len) * shadow_dist))

        raw_blur = MIN_SHADOW_BLUR + (MAX_SHADOW_BLUR - MIN_SHADOW_BLUR) * tilt_ratio
        blur_size = int(raw_blur)
        if blur_size % 2 == 0:
            blur_size += 1
        opacity = random.uniform(*SHADOW_OPACITY_RANGE)

    # 2. 그림자 생성 및 캔버스 합성 (커널 에로전 후 가우시안 블러로 소프트 섀도우 구현)
    erosion_ksize = max(3, int(min(h, w) * 0.06))
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_ksize, erosion_ksize))
    shadow_core_mask = cv2.erode(alpha_raw, erode_kernel)

    pad = blur_size * 2
    pad_shadow = np.zeros((h + pad * 2, w + pad * 2), dtype=np.uint8)
    pad_shadow[pad:pad + h, pad:pad + w] = shadow_core_mask

    shadow_soft_raw = cv2.GaussianBlur(pad_shadow.astype(np.float32), (blur_size, blur_size), 0)
    shadow_mask_final = (shadow_soft_raw / 255.0) * opacity

    sx, sy = x + offset_x - pad, y + offset_y - pad
    sx0, sy0 = max(0, sx), max(0, sy)
    sx1, sy1 = min(W, sx + w + pad * 2), min(H, sy + h + pad * 2)

    if sx1 > sx0 and sy1 > sy0:
        ssrc_x0, ssrc_y0 = sx0 - sx, sy0 - sy
        ssrc_x1, ssrc_y1 = ssrc_x0 + (sx1 - sx0), ssrc_y0 + (sy1 - sy0)
        cropped_shadow_mask = shadow_mask_final[ssrc_y0:ssrc_y1, ssrc_x0:ssrc_x1]
        shadow_mask_3ch = np.dstack([cropped_shadow_mask] * 3)
        s_roi = canvas_bgr[sy0:sy1, sx0:sx1].astype(np.float32)
        s_blended = s_roi * (1.0 - shadow_mask_3ch) + np.zeros_like(s_roi) * shadow_mask_3ch
        canvas_bgr[sy0:sy1, sx0:sx1] = s_blended.astype(np.uint8)

    # 3. 알약 본체 알파 블렌딩 처리 (경계면 페더링 적용으로 칼로 자른 듯한 경계 현상 제거)
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return canvas_bgr

    src_x0, src_y0 = x0 - x, y0 - y
    src_x1, src_y1 = src_x0 + (x1 - x0), src_y0 + (y1 - y0)
    rgb = rgba[src_y0:src_y1, src_x0:src_x1, :3]
    alpha_pill_raw = rgba[src_y0:src_y1, src_x0:src_x1, 3]

    if FEATHER_PX > 0:
        alpha_blur = cv2.GaussianBlur(alpha_pill_raw.astype(np.float32), (FEATHER_PX * 2 + 1, FEATHER_PX * 2 + 1), 0)
        alpha = np.minimum(alpha_pill_raw.astype(np.float32), alpha_blur) / 255.0
    else:
        alpha = alpha_pill_raw.astype(np.float32) / 255.0

    alpha_3ch = np.dstack([alpha] * 3)
    roi = canvas_bgr[y0:y1, x0:x1].astype(np.float32)
    blended = roi * (1 - alpha_3ch) + rgb.astype(np.float32) * alpha_3ch
    canvas_bgr[y0:y1, x0:x1] = blended.astype(np.uint8)
    return canvas_bgr


def match_illumination_soft(src_rgba: np.ndarray, target_bgr: np.ndarray, intensity: float = 0.3) -> np.ndarray:
    """[색감 조화] 주변 배경의 평균 밝기를 분석하여 알약의 밝기를 배경 스케일에 맞춰 은은하게 재조정"""
    alpha = src_rgba[:, :, 3]
    src_bgr = src_rgba[:, :, :3].astype(np.float32)
    target_gray = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2GRAY)
    bg_brightness = np.mean(target_gray)
    illumination_factor = bg_brightness / 128.0
    adjusted_factor = 1.0 + (illumination_factor - 1.0) * intensity
    res_bgr = src_bgr * adjusted_factor
    res_bgr = np.clip(res_bgr, 0, 255).astype(np.uint8)
    return np.dstack([res_bgr, alpha])


def optimize_and_shading_pill(rgba: np.ndarray):
    """33.3%의 확률로 알약에 무작위 각도의 사선 광원 연출을 가동"""
    active_light_angle = None
    if random.random() < 0.333:
        active_light_angle = random.uniform(0, 360)
        shading_intensity = random.uniform(0.16, 0.32)
        rgba = apply_directional_lighting(rgba, active_light_angle, max_shading=shading_intensity)
    return rgba, active_light_angle


def synthesize_one_image(pill_bank_by_cat: dict, bg_paths, canvas_size, chosen_cats, min_gap_px):
    """
        [단일 합성] 무작위 배경 이미지 1장을 선택해 지정된 카테고리의 알약들을
        스케일/회전/조명 변환을 준 뒤 비겹침 안전 영역 내에 배치해 1장의 이미지로 합성
    """
    cw, ch = canvas_size
    bg_path = random.choice(bg_paths)
    bg = cv2.imread(str(bg_path), cv2.IMREAD_COLOR)
    bg = cv2.resize(bg, (cw, ch))
    canvas = bg.copy()

    placed_boxes = []
    gt = []
    placed_cats_success = []
    shadow_safe_margin = (MAX_SHADOW_DIST // 2) + (MIN_GAP_PX)

    for cat_id in chosen_cats:
        if is_excluded_category(cat_id):
            continue

        if cat_id not in pill_bank_by_cat or not pill_bank_by_cat[cat_id]:
            continue

        rec = random.choice(pill_bank_by_cat[cat_id])
        rgba = cv2.imread(rec["rgba_path"], cv2.IMREAD_UNCHANGED)
        if rgba is None or rgba.shape[2] != 4:
            continue

        # 기하학적/광학적 변환 (조명, 회전, 스케일링)
        rgba, active_light_angle = optimize_and_shading_pill(rgba)
        angle = random.uniform(0, 360)
        scale = random.uniform(0.85, 1)
        rgba_t = rotate_rgba(rgba, angle)
        new_w, new_h = int(rgba_t.shape[1] * scale), int(rgba_t.shape[0] * scale)
        if new_w < 5 or new_h < 5 or new_w > cw or new_h > ch:
            continue
        rgba_t = cv2.resize(rgba_t, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # 이미지 테두리 밖으로 튕겨 나가지 않도록 배치 경계 설정
        x_min_placement = shadow_safe_margin
        x_max_placement = cw - new_w - shadow_safe_margin
        y_min_placement = shadow_safe_margin
        y_max_placement = ch - new_h - shadow_safe_margin

        if x_max_placement <= x_min_placement or y_max_placement <= y_min_placement:
            continue

        # MAX_PLACEMENT_TRIES(500회) 시도하며 비겹침 난수 좌표 탐색
        for _try in range(MAX_PLACEMENT_TRIES):
            x = random.randint(x_min_placement, x_max_placement)
            y = random.randint(y_min_placement, y_max_placement)
            box = (x, y, new_w, new_h)

            conflict = any(boxes_overlap_with_shadow_safety(box, pb, shadow_safe_margin) for pb in placed_boxes)
            if not conflict:
                # 배경 색조 매칭 후 마스크/그림자 융합 합성
                roi = canvas[y:y + new_h, x:x + new_w]
                if roi.shape[:2] == rgba_t.shape[:2]:
                    rgba_t = match_illumination_soft(rgba_t, roi, intensity=0.25)

                canvas = paste_rgba_with_smart_shadow(canvas, rgba_t, x, y, custom_light_angle=active_light_angle)
                placed_boxes.append(box)
                gt.append({
                    "category_id": rec["category_id"],
                    "category_name": rec["category_name"],
                    "bbox": [x, y, new_w, new_h],
                })
                placed_cats_success.append(cat_id)
                break

    return canvas, gt, placed_cats_success


def build_balanced_synthetic_dataset(pill_bank_by_cat, bg_paths, out_image_dir, out_annot_path,
                                     target_instances, pills_per_image_range, canvas_size, min_gap_px):
    """
        [대형 루프 및 밸런싱] 목표 개수가 소진될 때까지 무작위 알약 주머니(pill_bag)에서 알약을 추출 및 조합하여
        데이터 불균형이 해소된 데이터셋 빌드
    """
    if os.path.exists(out_image_dir):
        shutil.rmtree(out_image_dir)
    os.makedirs(out_image_dir, exist_ok=True)

    coco = {"images": [], "annotations": [], "categories": []}
    cat_seen = {}
    ann_id = 1

    # 제외 클래스(다수 클래스)를 제외한 소수 클래스 리스트업
    available_cats = [c for c in pill_bank_by_cat.keys() if not is_excluded_category(c) and pill_bank_by_cat[c]]
    if not available_cats:
        print("⚠️ [경고] 합성에 사용할 수 있는 소수 클래스 자산이 없습니다.")
        return

    # 각 클래스당 목표 인스턴스(300개)만큼 균등하게 적재 후 셔플
    pill_bag = []
    for cat in available_cats:
        pill_bag.extend([cat] * target_instances)
    random.shuffle(pill_bag)

    print(f"\n--- [시작] 총 {len(pill_bag)}개의 객체 대상 1:2 사선 광원/그림자 믹스매치 증강 합성 시작 ---")

    img_id = 1
    with tqdm(total=len(pill_bag), desc="목표 객체 생성 완료율") as pbar:
        while pill_bag:
            n_pills = random.randint(*pills_per_image_range)
            n_pills = min(n_pills, len(set(pill_bag)))
            if n_pills == 0:
                break

            chosen_cats = []
            temp_held = []

            # 한 장의 이미지 안에서 동일한 알약이 중복 배치되지 않도록 고유 샘플링
            while len(chosen_cats) < n_pills and pill_bag:
                drawn = pill_bag.pop()
                if drawn in chosen_cats:
                    temp_held.append(drawn)
                else:
                    chosen_cats.append(drawn)

            if temp_held:
                pill_bag.extend(temp_held)
                random.shuffle(pill_bag)

            if not chosen_cats:
                break

            canvas, gt, placed_cats_success = synthesize_one_image(
                pill_bank_by_cat, bg_paths, canvas_size, chosen_cats, min_gap_px
            )

            # 배치 공간 부족 등의 사유로 배치에 실패한 알약은 주머니에 되돌려 놓음
            failed_cats = set(chosen_cats) - set(placed_cats_success)
            if failed_cats:
                pill_bag.extend(list(failed_cats))
                random.shuffle(pill_bag)

            if not gt:
                continue

            # 이미지 저장 및 COCO 데이터 정의 등록
            file_name = f"syn_{img_id:06d}.png"
            cv2.imwrite(os.path.join(out_image_dir, file_name), canvas)
            coco["images"].append({
                "id": img_id, "file_name": file_name,
                "width": canvas_size[0], "height": canvas_size[1],
            })

            for g in gt:
                cid = g["category_id"]
                cat_seen.setdefault(cid, g["category_name"])
                coco["annotations"].append({
                    "id": ann_id, "image_id": img_id, "category_id": cid,
                    "bbox": g["bbox"], "area": g["bbox"][2] * g["bbox"][3], "iscrowd": 0,
                })
                ann_id += 1

            pbar.update(len(placed_cats_success))
            img_id += 1

    # 최종 COCO json 라벨 파일 저장
    coco["categories"] = [{"id": cid, "name": name, "supercategory": "pill"} for cid, name in cat_seen.items()]
    with open(out_annot_path, "w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False, indent=2)

    print(f"\n[완료] 총 {img_id - 1}장의 하이브리드 조명 데이터 합성이 완료되었습니다.")


def main():
    print("=== 1. 정제 완료된 YOLO .txt 기반 정보 수집 ===")
    merged = collect_annotations_from_yolo_txt()

    pill_bank = []

    # 기존에 추출해 둔 자산 창고가 있다면 중복으로 SAM 연산을 하지 않음
    if os.path.exists(PILL_BANK_DIR) and len(os.listdir(PILL_BANK_DIR)) > 0:
        print("\n⚡ [캐시 전수조사 스캔] 마스터 Pill Bank 내부의 모든 자산을 적재합니다.")
        for f in os.listdir(PILL_BANK_DIR):
            if not f.lower().endswith('.png'):
                continue

            match = re.search(r"_cat(\d+)\.png", f)
            if match:
                try:
                    cid = int(match.group(1))
                    pill_bank.append({
                        "rgba_path": os.path.join(PILL_BANK_DIR, f),
                        "category_id": cid,
                        "category_name": CLASS_NAMES.get(cid, f"Class_{cid}")
                    })
                except Exception:
                    continue

    # 만약 기존 캐시에 55번 초과 데이터가 발견되지 않았거나 비어있다면 새 빌드 가동
    cached_max_id = max([r["category_id"] for r in pill_bank]) if pill_bank else 0
    if not pill_bank or cached_max_id <= 55:
        print("\n🔄 [캐시 무효화 및 갱신] 93개 클래스 전수 저장을 위해 마스터 마스크를 처음부터 다시 추출합니다.")
        if os.path.exists(PILL_BANK_DIR):
            shutil.rmtree(PILL_BANK_DIR)
        os.makedirs(SYN_IMAGE_DIR, exist_ok=True)
        pill_bank = build_pill_bank_with_sam(merged, IMAGE_DIR, PILL_BANK_DIR)

    if not pill_bank:
        raise RuntimeError("pill bank 데이터셋이 비었습니다.")

    pill_bank_all_debug = defaultdict(list)
    pill_bank_by_cat = defaultdict(list)

    for rec in pill_bank:
        cid = rec["category_id"]
        pill_bank_all_debug[cid].append(rec)

        if is_excluded_category(cid):
            continue
        pill_bank_by_cat[cid].append(rec)

    print("\n📦 [Pill Bank 다중 자산 구축 리포트 - 전수조사 결과]")
    print("=" * 75)
    print(f"{'클래스 ID':<8} | {'알약 제품명':<35} | {'누끼 확보 수':<10} | {'상태'}")
    print("-" * 75)

    for cid in range(93):
        count = len(pill_bank_all_debug[cid])
        pill_name = CLASS_NAMES.get(cid, 'Unknown')

        if is_excluded_category(cid):
            status = "[제외 대상 (고정)]"
        elif count == 0:
            status = "[⚠️ 데이터 공백]"
        else:
            status = "[증강 대상]"

        print(f"ID {cid:<5} | {pill_name:<40} | {count:<12} | {status}")
    print("=" * 75)

    print(f"[INFO] 최종 합성에 투입할 소수 카테고리 수: {len(pill_bank_by_cat)} (다수 클래스 배제 완료)")

    print("\n=== 3. 사용자 제공 배경 로드 ===")
    bg_paths = load_user_backgrounds(BG_PATHS)

    if not bg_paths:
        raise RuntimeError(f"정상적인 배경 이미지(.png)를 찾을 수 없습니다. BG_PATHS 경로를 확인해 주세요.")

    print("\n=== 4. 무작위 완전 비겹침 대량 합성 ===")
    build_balanced_synthetic_dataset(
        pill_bank_by_cat, bg_paths, SYN_IMAGE_DIR, SYN_ANNOT_PATH,
        TARGET_INSTANCES_PER_CAT, PILLS_PER_IMAGE_RANGE, CANVAS_SIZE, MIN_GAP_PX
    )


if __name__ == "__main__":
    main()