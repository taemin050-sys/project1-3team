"""학습 데이터와 Kaggle 테스트셋(조합2) 간 leakage(컨닝) 감사.

테스트 이미지는 정수(1.png)로 익명화돼 파일명으로는 대조가 불가능하므로,
**이미지 내용**으로 대조한다:

1. 정확 픽셀 해시(sha1 of RGB pixels): 동일 사진이면 확실한 leakage.
2. 지각 해시(dHash, Hamming≤threshold): 리사이즈/재저장된 동일 사진 탐지.
3. 구조 확인: AIHub에서 조합2(TS_2)를 실제로 안 썼는지.

사용법(맥에서, 프로젝트 루트에서):
    python -m src.joelchoi.audit_leakage --combos 1 3
또는 노트북에서:
    from src.joelchoi.audit_leakage import audit
    audit(combos=[1, 3])
"""

import argparse
import hashlib
from pathlib import Path

from PIL import Image

KAGGLE_DATA = (
    Path.home()
    / ".cache/kagglehub/competitions/ai12-level1-project/sprint_ai_project1_data"
)


def _pixel_sha1(path: Path) -> str | None:
    try:
        with Image.open(path) as im:
            return hashlib.sha1(im.convert("RGB").tobytes()).hexdigest()
    except Exception:
        return None


def _dhash(path: Path, size: int = 8) -> int | None:
    """8x8 dHash(64-bit). 리사이즈/재인코딩에 강건."""
    try:
        with Image.open(path) as im:
            g = im.convert("L").resize((size + 1, size), Image.BILINEAR)
        px = list(g.get_flattened_data())
        bits = 0
        for r in range(size):
            row = px[r * (size + 1) : (r + 1) * (size + 1)]
            for c in range(size):
                bits = (bits << 1) | int(row[c] < row[c + 1])
        return bits
    except Exception:
        return None


def _gray_vec(path: Path, n: int = 64) -> list[float] | None:
    """n×n 그레이스케일 정규화 벡터(정밀 픽셀 대조용). 같은 사진이면 RMSE≈0."""
    try:
        with Image.open(path) as im:
            g = im.convert("L").resize((n, n), Image.BILINEAR)
        return [v / 255.0 for v in g.getdata()]
    except Exception:
        return None


def _rmse(a: list[float], b: list[float]) -> float:
    return (sum((x - y) ** 2 for x, y in zip(a, b)) / len(a)) ** 0.5


def _collect_train_images(combos: list[int], include_vs: bool = False) -> list[Path]:
    """Kaggle train_images + AIHub 사용 조합 이미지(+옵션 VS) 경로 수집."""
    paths = [
        p for p in (KAGGLE_DATA / "train_images").glob("*.png")
        if "_index" not in p.name
    ]

    try:
        from src.joelchoi.data.aihub_converter import (
            EXCLUDED_COMBOS,
            _collect_image_dir,
            find_aihub_root,
        )

        assert 2 not in combos, "조합2는 테스트셋 — 학습 사용 금지"
        assert not (set(combos) & EXCLUDED_COMBOS), "제외 조합 포함됨"
        root = find_aihub_root()
        image_dirs = _collect_image_dir(root)
        for c in combos:
            d = image_dirs.get(c)
            if d:
                paths.extend(p for p in d.rglob("*.png") if "_index" not in p.name)

        # AIHub 공식 Validation(VS) — 테스트가 여기서 파생됐는지 확인용
        if include_vs:
            vs_root = root.parent / "2.Validation" / "원천데이터" / "경구약제조합 5000종"
            if vs_root.exists():
                paths.extend(
                    p for p in vs_root.rglob("*.png") if "_index" not in p.name
                )
            else:
                print(f"[경고] VS 경로 없음: {vs_root}")
    except Exception as e:
        print(f"[경고] AIHub 이미지 수집 생략: {e}")
    return paths


def audit(
    combos: list[int] | None = None,
    hash_size: int = 16,
    hash_cutoff: int = 12,
    rmse_cutoff: float = 0.06,
    include_vs: bool = False,
) -> dict:
    """정밀 2단계 leakage 감사.

    1) 정확 픽셀 해시 → 동일 파일(확실한 leakage) 탐지.
    2) 256-bit(16×16) 지각해시로 각 테스트 이미지의 '최근접 학습 이미지'를 찾고,
       해시 거리 ≤ hash_cutoff 후보만 **32×32 grayscale RMSE**로 재검증.
       RMSE ≤ rmse_cutoff 이면 사실상 같은 사진(진짜 의심), 그 이상은 '구도만 비슷'.

    이렇게 하면 배경·트레이 구도가 같아 생기는 오탐을 걸러낸다.
    """
    combos = combos or [1, 3]
    print(f"=== Leakage 감사 (AIHub 조합 {combos} + Kaggle train vs 테스트셋) ===")
    print("구조 확인: 조합2 사용?", "예 (문제!)" if 2 in combos else "아니오 (정상)")

    test_paths = sorted((KAGGLE_DATA / "test_images").glob("*.png"))
    train_paths = _collect_train_images(combos, include_vs=include_vs)
    print(f"테스트 {len(test_paths)}장, 학습 후보 {len(train_paths)}장 해싱 중...")
    if include_vs:
        print("※ VS(AIHub 공식 Validation) 포함 대조 — 테스트가 VS 파생인지 확인")

    # 1) 정확 픽셀 해시 + 256-bit 지각해시
    train_px: dict[str, Path] = {}
    train_dh: list[tuple[int, Path]] = []
    for p in train_paths:
        h = _pixel_sha1(p)
        if h:
            train_px.setdefault(h, p)
        d = _dhash(p, size=hash_size)
        if d is not None:
            train_dh.append((d, p))

    exact_hits = []
    hist = {}  # 최근접 해시거리 분포
    candidates = []  # (test_path, best_train_path, hash_dist)
    for tp in test_paths:
        th = _pixel_sha1(tp)
        if th and th in train_px:
            exact_hits.append((tp, train_px[th]))
            continue
        td = _dhash(tp, size=hash_size)
        if td is None:
            continue
        best_d, best_p = min(
            ((bin(td ^ d).count("1"), p) for d, p in train_dh),
            key=lambda x: x[0],
        )
        bucket = min(best_d // 4 * 4, 40)  # 0-3,4-7,... 로 묶기
        hist[bucket] = hist.get(bucket, 0) + 1
        if best_d <= hash_cutoff:
            candidates.append((tp, best_p, best_d))

    # 2) 후보 정밀 재검증(RMSE)
    train_vec_cache: dict[Path, list[float] | None] = {}
    near_dups = []
    for tp, sp, hd in candidates:
        tv = _gray_vec(tp)
        if sp not in train_vec_cache:
            train_vec_cache[sp] = _gray_vec(sp)
        sv = train_vec_cache[sp]
        if tv is None or sv is None:
            continue
        r = _rmse(tv, sv)
        if r <= rmse_cutoff:
            near_dups.append((tp, sp, hd, r))

    print("\n── 결과 ──────────────────────────────")
    print(f"정확 픽셀 일치(확실한 leakage): {len(exact_hits)}")
    for t, s in exact_hits[:5]:
        print(f"   TEST {t.name}  ==  TRAIN {s.name}")

    print("\n최근접 학습이미지까지의 해시거리 분포(작을수록 유사):")
    for b in sorted(hist):
        print(f"   {b:2d}-{b + 3:2d} bit: {hist[b]:4d}장")

    # 출처 분류: Kaggle train / AIHub TS(우리증강) / VS(공식 Validation)
    def _source(p: Path) -> str:
        s = str(p)
        if "/2.Validation/" in s or "VS_" in s:
            return "VS"
        if "/train_images/" in s or "train_images" in Path(s).parts:
            return "kaggle"
        return "aihub"

    n_kaggle = sum(1 for _, s, _, _ in near_dups if _source(s) == "kaggle")
    n_aihub = sum(1 for _, s, _, _ in near_dups if _source(s) == "aihub")
    n_vs = sum(1 for _, s, _, _ in near_dups if _source(s) == "VS")

    print(
        f"\n해시근접(≤{hash_cutoff}) 후보 {len(candidates)}장 중 "
        f"RMSE≤{rmse_cutoff}(사실상 동일 사진): {len(near_dups)}장"
    )
    print(
        f"   └ 출처별: Kaggle train {n_kaggle}, AIHub TS(우리증강) {n_aihub}, "
        f"VS(공식 Validation) {n_vs}"
    )
    for t, s, hd, r in sorted(near_dups, key=lambda x: x[3])[:12]:
        print(
            f"   [{_source(s):6s}] TEST {t.name}  ~~  {s.name}  (hash {hd}, rmse {r:.3f})"
        )

    print("\n해석:")
    print("  · Kaggle 근접 = 대회 제공 train/test 내재적 중복 → baseline에도 포함")
    print("  · AIHub TS 근접 = 우리 증강이 넣은 테스트 근접 이미지")
    print("  · VS 근접 = 테스트가 VS 파생이라는 신호 → VS를 train/val로 쓰면 컨닝!")

    if n_vs > 0:
        verdict = (
            f"위험(VS 사용 금지) — 테스트가 VS와 {n_vs}장 겹침. "
            "VS를 train/val로 쓰면 실제 컨닝."
        )
    elif n_aihub > 0:
        verdict = f"검토 — AIHub TS 증강발 근접 {n_aihub}장. 해당 조합 제외 고려"
    elif near_dups:
        verdict = f"OK — 근접 {len(near_dups)}장 전부 Kaggle 제공 데이터 내재 중복"
    else:
        verdict = "OK — 동일 사진 없음"
    print(f"\n판정: {verdict}")
    return {
        "exact": exact_hits,
        "near_dups": near_dups,
        "n_near_kaggle": n_kaggle,
        "n_near_aihub": n_aihub,
        "n_near_vs": n_vs,
        "hist": hist,
        "n_test": len(test_paths),
        "n_train": len(train_paths),
        "verdict": verdict,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--combos", type=int, nargs="+", default=[1, 3])
    ap.add_argument("--hash-size", type=int, default=16)
    ap.add_argument("--hash-cutoff", type=int, default=12)
    ap.add_argument("--rmse-cutoff", type=float, default=0.06)
    ap.add_argument(
        "--check-vs",
        action="store_true",
        help="AIHub 공식 Validation(VS)을 대조에 포함 — 테스트가 VS 파생인지 확인",
    )
    a = ap.parse_args()
    audit(
        combos=a.combos,
        hash_size=a.hash_size,
        hash_cutoff=a.hash_cutoff,
        rmse_cutoff=a.rmse_cutoff,
        include_vs=a.check_vs,
    )
