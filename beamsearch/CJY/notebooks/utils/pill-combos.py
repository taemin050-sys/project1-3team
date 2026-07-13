#!/usr/bin/env python3
"""
경구약 조합 데이터셋(TL_*_조합)을 pill_yolo의 한글약제명 클래스 기준으로 분석/필터링.

매칭 키: 한글약제명(정규화). pill_yolo data.yaml의 names(한글) vs TL JSON의 dl_name.

[모드 1] 조합(combination) 단위 필터  — 기본
  겹침비율(조합)=(조합 내 알약 중 타깃 클래스 수)/(조합 내 알약 수)
  threshold 이하 조합을 _excluded/로 이동(--apply). 기본은 dry-run.

[모드 2] 데이터셋(zip) 단위 리포트 — --by-dataset
  각 데이터셋(상위 폴더)이 타깃 56클래스와 얼마나 겹치는지, 학습에 쓸 수 있는
  조합이 몇 개인지 표로 출력하고, DOWNLOAD/SKIP 권고를 준다.

사용:
  # 데이터셋 단위 판단 (TL_1/, TL_3/ ... 가 들어있는 상위 폴더를 --tl-root 로)
  python filter_pill_combos.py --tl-root <상위폴더> --pill-yolo <pill_yolo> \
      --by-dataset --threshold 0.2 --by-dataset-csv dataset_report.csv

  # 조합 단위 필터 (dry-run → 확인 후 --apply)
  python filter_pill_combos.py --tl-root <폴더> --pill-yolo <pill_yolo> --threshold 0.2
  python filter_pill_combos.py ... --threshold 0.2 --apply
"""

import os
import re
import csv
import json
import shutil
import argparse
from collections import Counter

CODE_RE = re.compile(r"K-\d{6}")


def normalize_name(s: str) -> str:
    """한글약제명 정규화: 괄호 제거 → 슬래시 이후 제거 → 공백 제거.
    '게보린정 300mg/PTP' -> '게보린정300mg' / '마그밀정(수산화마그네슘)' -> '마그밀정'
    """
    s = s or ""
    s = re.sub(r"\(.*?\)", "", s)
    s = re.sub(r"\[.*?\]", "", s)
    s = re.sub(r"/.*$", "", s)
    s = re.sub(r"\s+", "", s)
    return s


def load_yolo_names(pill_yolo_dir: str):
    yaml_path = None
    for cand in ("data.yaml", "data.yml", "dataset.yaml"):
        p = os.path.join(pill_yolo_dir, cand)
        if os.path.exists(p):
            yaml_path = p
            break
    if not yaml_path:
        raise FileNotFoundError(f"{pill_yolo_dir} 에서 data.yaml을 못 찾음")
    import yaml

    with open(yaml_path, encoding="utf-8") as f:
        d = yaml.safe_load(f)
    n = d.get("names")
    if isinstance(n, dict):
        names = [str(v) for v in n.values()]
    elif isinstance(n, list):
        names = [str(v) for v in n]
    else:
        raise ValueError("data.yaml names 형식을 해석 못함")
    return names


def list_combo_dirs(root: str):
    return [
        os.path.join(root, d)
        for d in sorted(os.listdir(root))
        if d.endswith("_json") and os.path.isdir(os.path.join(root, d))
    ]


def iter_datasets(tl_root: str):
    """(dataset_name, [combo_dir, ...]) 를 yield.
    tl_root 바로 아래에 *_json 이 있으면 그 자체를 단일 데이터셋으로 취급,
    아니면 한 단계 아래 각 하위폴더(TL_1, TL_3 ...)를 데이터셋으로 취급.
    """
    direct = list_combo_dirs(tl_root)
    if direct:
        yield os.path.basename(os.path.normpath(tl_root)), direct
        return
    for t in sorted(os.listdir(tl_root)):
        tp = os.path.join(tl_root, t)
        if os.path.isdir(tp):
            combos = list_combo_dirs(tp)
            if combos:
                yield t, combos


def all_combos(tl_root: str):
    out = []
    for ds_name, combos in iter_datasets(tl_root):
        for c in combos:
            out.append((os.path.basename(c), c, ds_name))
    return out


def build_code2name(combo_dirs):
    code2name = {}
    for cdir in combo_dirs:
        for sub in os.listdir(cdir):
            sdir = os.path.join(cdir, sub)
            if not os.path.isdir(sdir):
                continue
            m = CODE_RE.match(sub)
            if not m:
                continue
            code = m.group(0)
            if code in code2name:
                continue
            js = [f for f in os.listdir(sdir) if f.endswith(".json")]
            if not js:
                continue
            try:
                j = json.load(open(os.path.join(sdir, js[0]), encoding="utf-8"))
                code2name[code] = j["images"][0].get("dl_name")
            except Exception:
                pass
    return code2name


def combo_codes(combo_dir: str):
    out = set()
    for s in os.listdir(combo_dir):
        if os.path.isdir(os.path.join(combo_dir, s)) and s.startswith("K-"):
            m = CODE_RE.match(s)
            out.add(m.group(0) if m else s)
    return out


def diagnose_matching(code2name, yolo_norm):
    tl_norm = {normalize_name(v) for v in code2name.values()}
    matched = [orig for nm, orig in yolo_norm.items() if nm in tl_norm]
    unmatched = [orig for nm, orig in yolo_norm.items() if nm not in tl_norm]
    print(
        f"\n[매칭 진단] 내 클래스 중 TL과 매칭 {len(matched)} / 미매칭 {len(unmatched)}"
    )
    if unmatched:
        print("  ⚠ TL에 없는(또는 표기 불일치) 클래스 — 직접 확인 권장:")
        for u in unmatched:
            print(f"     - {u}")


# ---------- 모드 2: 데이터셋(zip) 단위 ----------
def run_by_dataset(
    tl_root, code2name, yolo_norm, threshold, csv_path=None, metric="usable"
):
    # 권고 기준 metric: 'usable'(타깃 포함 조합 비율, 권장) / 'ds_overlap' / 'tgt_cov'
    metric_idx = {"usable": 7, "ds_overlap": 3, "tgt_cov": 4}[metric]
    n_targets = len(yolo_norm)
    rows = []
    for ds_name, combos in iter_datasets(tl_root):
        ds_codes = set()
        usable = 0
        for c in combos:
            cc = combo_codes(c)
            ds_codes |= cc
            names_norm = [normalize_name(code2name.get(x, "")) for x in cc]
            if any(n in yolo_norm for n in names_norm):
                usable += 1
        ds_names_norm = {normalize_name(code2name.get(x, "")) for x in ds_codes}
        matched = sum(1 for n in ds_names_norm if n in yolo_norm)
        n_cls = len(ds_codes)
        ds_overlap = (
            matched / n_cls if n_cls else 0.0
        )  # 이 데이터셋 클래스 중 타깃 비율
        tgt_cov = (
            matched / n_targets if n_targets else 0.0
        )  # 내 56개 중 이 데이터셋이 커버
        usable_ratio = usable / len(combos) if combos else 0.0
        row = [
            ds_name,
            n_cls,
            matched,
            ds_overlap,
            tgt_cov,
            len(combos),
            usable,
            usable_ratio,
            None,
        ]
        row[8] = "SKIP" if row[metric_idx] <= threshold else "DOWNLOAD"
        rows.append(row)

    # 출력
    print(
        f"\n=== 데이터셋(zip) 단위 겹침 리포트  (타깃 {n_targets}클래스, "
        f"기준={metric}, threshold={threshold}) ==="
    )
    hdr = [
        "dataset",
        "cls",
        "match",
        "ds_ovl%",
        "tgt_cov%",
        "combos",
        "usable",
        "usable%",
        "권고",
    ]
    print("  {:<14}{:>5}{:>7}{:>9}{:>10}{:>8}{:>8}{:>9}  {}".format(*hdr))
    for r in rows:
        print(
            "  {:<14}{:>5}{:>7}{:>8.0%}{:>10.0%}{:>8}{:>8}{:>9.0%}  {}".format(
                r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8]
            )
        )

    dl = [r[0] for r in rows if r[8] == "DOWNLOAD"]
    sk = [r[0] for r in rows if r[8] == "SKIP"]
    print(f"\n  → DOWNLOAD {len(dl)}개: {dl}")
    print(f"  → SKIP     {len(sk)}개: {sk}")
    print(f"\n  * 권고 기준 = {metric}  (--by-dataset-metric 로 변경 가능)")
    print("  * usable%  = 타깃 1개 이상 포함해 학습에 쓸 수 있는 조합 비율 (권장 기준)")
    print("  * ds_ovl%  = 이 데이터셋 클래스 중 내 타깃 비율")
    print("  * tgt_cov% = 내 타깃 56개 중 이 데이터셋이 커버하는 비율")

    if csv_path:
        with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "dataset",
                "classes",
                "matched",
                "ds_overlap",
                "target_coverage",
                "combos",
                "usable_combos",
                "usable_ratio",
                "recommendation",
            ])
            for r in rows:
                w.writerow([
                    r[0],
                    r[1],
                    r[2],
                    f"{r[3]:.4f}",
                    f"{r[4]:.4f}",
                    r[5],
                    r[6],
                    f"{r[7]:.4f}",
                    r[8],
                ])
        print(f"\n  CSV 저장: {csv_path}")


# ---------- 모드 1: 조합 단위 ----------
def run_by_combo(
    combos_all,
    code2name,
    yolo_norm,
    threshold,
    tl_root,
    apply,
    excluded_dir,
    report_csv,
):
    keep, drop = [], []
    hist = Counter()
    rows = []
    for name, path, ds in combos_all:
        codes = combo_codes(path)
        if not codes:
            continue
        names_norm = [normalize_name(code2name.get(c, "")) for c in codes]
        inter = [n for n in names_norm if n in yolo_norm]
        ratio = len(inter) / len(codes)
        hist[round(ratio, 2)] += 1
        rec = (name, path, ratio, len(inter), len(codes))
        (drop if ratio <= threshold else keep).append(rec)
        rows.append([
            ds,
            name,
            f"{ratio:.3f}",
            len(inter),
            len(codes),
            "DROP" if ratio <= threshold else "KEEP",
        ])

    print(
        f"\n총 조합 {len(combos_all)} | 유지 {len(keep)} | 제외 {len(drop)}  (threshold={threshold})"
    )
    print("겹침비율 분포:", dict(sorted(hist.items())))

    if report_csv:
        with open(report_csv, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["dataset", "combo", "ratio", "matched", "total", "decision"])
            w.writerows(rows)
        print(f"리포트 저장: {report_csv}")

    if apply:
        out = excluded_dir or os.path.join(tl_root, "_excluded")
        os.makedirs(out, exist_ok=True)
        moved = 0
        for name, path, *_ in drop:
            dst = os.path.join(out, name)
            if os.path.exists(dst):
                continue
            shutil.move(path, dst)
            moved += 1
        print(f"\n✓ 제외 {moved}개 조합을 {out} 로 이동 완료.")
    else:
        print("\n(dry-run) 실제 이동하려면 --apply 추가.")
        for name, _, r, i, t in drop[:10]:
            print(f"  DROP {name}  ratio={r:.2f} ({i}/{t})")
        if len(drop) > 10:
            print(f"  ... 외 {len(drop) - 10}개")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tl-root", required=True, help="TL 조합 폴더들의 상위 경로")
    ap.add_argument("--pill-yolo", required=True, help="pill_yolo 데이터셋 경로")
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.2,
        help='이 값 "이하" 겹침이면 제외/SKIP (기본 0.2)',
    )
    # 모드 2
    ap.add_argument(
        "--by-dataset",
        action="store_true",
        help="데이터셋(zip) 단위 겹침 리포트 후 종료",
    )
    ap.add_argument(
        "--by-dataset-metric",
        default="usable",
        choices=["usable", "ds_overlap", "tgt_cov"],
        help="DOWNLOAD/SKIP 권고 기준 (기본 usable)",
    )
    ap.add_argument("--by-dataset-csv", default=None)
    # 모드 1
    ap.add_argument("--apply", action="store_true", help="조합 제외(이동) 실제 수행")
    ap.add_argument("--excluded-dir", default=None)
    ap.add_argument("--report-csv", default=None)
    args = ap.parse_args()

    yolo_names = load_yolo_names(args.pill_yolo)
    yolo_norm = {normalize_name(n): n for n in yolo_names}
    print(f"[pill_yolo] 클래스 {len(yolo_names)}개 (정규화 후 고유 {len(yolo_norm)}개)")

    combos_all = all_combos(args.tl_root)
    code2name = build_code2name([c for _, c, _ in combos_all])
    print(f"[TL] 조합 {len(combos_all)}개 | 약제 K-code {len(code2name)}종")
    diagnose_matching(code2name, yolo_norm)

    if args.by_dataset:
        run_by_dataset(
            args.tl_root,
            code2name,
            yolo_norm,
            args.threshold,
            args.by_dataset_csv,
            args.by_dataset_metric,
        )
    else:
        run_by_combo(
            combos_all,
            code2name,
            yolo_norm,
            args.threshold,
            args.tl_root,
            args.apply,
            args.excluded_dir,
            args.report_csv,
        )


if __name__ == "__main__":
    main()
