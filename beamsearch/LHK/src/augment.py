"""단일→조합 증강 엔진 (LHK). Copy-Paste Synthesis: (2) AI Hub 단일 → (1) 도메인 조합.
설계: augmentation_design.md. (2) 도착 전에도 (1) 데이터로 엔진 검증(smoke_augment.py) 가능.

핵심 함수:
  wb_gray_world  : WB 정규화(주백색 톤, 알약 영역 기준)
  segment_pill   : 알약 분할(CV 휘도 Otsu; SAM은 lazy 스텁)
  cutout         : 마스크→RGBA 컷아웃(+경계 침식)
  sample_background: (1) 빈 연회색 영역 크롭 배경
  compose        : 비겹침 그리드 배치 + 알파 합성 → (이미지, [(cat,[x,y,w,h])])
"""

from __future__ import annotations
import random
import numpy as np
import cv2


# ---------- WB 정규화 ----------
def wb_gray_world(rgb: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """gray-world: 알약 영역(mask>0) 평균이 무채색이 되도록 채널 게인. (1) 주백색 톤 정합."""
    region = rgb[mask > 0] if mask is not None else rgb.reshape(-1, 3)
    if region.size == 0:
        return rgb
    mean = region.reshape(-1, 3).mean(0) + 1e-6
    gain = mean.mean() / mean
    return np.clip(rgb.astype(np.float32) * gain, 0, 255).astype(np.uint8)


# ---------- 분할 ----------
def segment_pill_cv(rgb: np.ndarray) -> np.ndarray | None:
    """휘도 Otsu + 배경극성 자동판정 + 모폴로지 + 최대 컨투어. 불투명 알약용."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    k = max(4, min(gray.shape) // 20)
    corners = np.concatenate(
        [
            gray[:k, :k].ravel(),
            gray[:k, -k:].ravel(),
            gray[-k:, :k].ravel(),
            gray[-k:, -k:].ravel(),
        ]
    )
    bg_bright = corners.mean() > gray.mean()  # 배경이 알약보다 밝은가
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if bg_bright:
        th = 255 - th  # 알약이 어두우면 전경 반전
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [max(cnts, key=cv2.contourArea)], -1, 255, -1)
    return mask


def segment_pill(rgb: np.ndarray, use_sam: bool = False) -> np.ndarray | None:
    """CV 우선. use_sam=True는 어려운 케이스(반투명 등) — segment-anything 필요(Colab 배치 권장)."""
    if use_sam:
        raise NotImplementedError("SAM 경로 미구현: segment-anything + 체크포인트 필요")
    return segment_pill_cv(rgb)


# ---------- 컷아웃 ----------
def cutout(rgb: np.ndarray, mask: np.ndarray, erode_px: int = 2) -> np.ndarray | None:
    """마스크 경계 침식(fringe 제거) 후 최소 bbox로 크롭한 RGBA 반환."""
    if mask is None:
        return None
    if erode_px > 0:
        mask = cv2.erode(mask, np.ones((erode_px * 2 + 1,) * 2, np.uint8))
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return None
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    crop = rgb[y0 : y1 + 1, x0 : x1 + 1]
    m = mask[y0 : y1 + 1, x0 : x1 + 1]
    return np.dstack([crop, m])


# ---------- 배경 ----------
def sample_background(
    bg_pool: list, size=(976, 1280), rng: random.Random | None = None
) -> np.ndarray:
    """(1) train 이미지의 빈 연회색 영역(상단 밴드)을 크롭·리사이즈해 배경 생성. size=(W,H)."""
    rng = rng or random
    W, H = size
    img = cv2.cvtColor(cv2.imread(str(rng.choice(bg_pool))), cv2.COLOR_BGR2RGB)
    h = img.shape[0]
    band = img[: max(1, h // 5)]  # 상단 빈 영역(알약 대개 중앙/하단)
    return cv2.resize(band, (W, H), interpolation=cv2.INTER_LINEAR)


# ---------- 기하증강 ----------
def _rotate_rgba(rgba: np.ndarray, deg: float) -> np.ndarray:
    h, w = rgba.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(
        rgba, M, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0)
    )


# ---------- 합성 (비겹침 그리드) ----------
def compose(
    bg: np.ndarray,
    pills: list,
    min_n=2,
    max_n=4,
    cell_fill=(0.55, 0.8),
    rng: random.Random | None = None,
):
    """pills=[(rgba, category_id)]. 2x2 그리드 셀에 1개씩(비겹침) 배치·회전·합성.
    반환: (합성 RGB, [(category_id, [x,y,w,h])]). 앞면 유지 위해 flip 없음."""
    rng = rng or random
    canvas = bg.copy()
    H, W = canvas.shape[:2]
    cols = rows = 2
    n = min(rng.randint(min_n, max_n), len(pills), cols * rows)
    chosen = rng.sample(pills, n)
    cells = [(c, r) for r in range(rows) for c in range(cols)]
    rng.shuffle(cells)
    anns = []
    for (rgba, cat), (c, r) in zip(chosen, cells):
        cw, ch = W // cols, H // rows
        s = min(cw, ch) * rng.uniform(*cell_fill) / max(rgba.shape[:2])
        rp = cv2.resize(
            rgba, (max(1, int(rgba.shape[1] * s)), max(1, int(rgba.shape[0] * s)))
        )
        rp = _rotate_rgba(rp, rng.uniform(0, 360))
        ph, pw = rp.shape[:2]
        pw, ph = min(pw, cw), min(ph, ch)
        rp = rp[:ph, :pw]
        x0 = c * cw + rng.randint(0, max(1, cw - pw))
        y0 = r * ch + rng.randint(0, max(1, ch - ph))
        a = rp[:, :, 3:4].astype(np.float32) / 255
        canvas[y0 : y0 + ph, x0 : x0 + pw] = (
            rp[:, :, :3] * a + canvas[y0 : y0 + ph, x0 : x0 + pw] * (1 - a)
        ).astype(np.uint8)
        ys, xs = np.where(rp[:, :, 3] > 10)
        if xs.size:
            anns.append(
                (
                    cat,
                    [
                        int(x0 + xs.min()),
                        int(y0 + ys.min()),
                        int(xs.max() - xs.min() + 1),
                        int(ys.max() - ys.min() + 1),
                    ],
                )
            )
    return canvas, anns
