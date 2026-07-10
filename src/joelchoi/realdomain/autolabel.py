"""오토라벨 보조: 현재 YOLO로 실촬영 이미지에 박스 초안(YOLO txt) 생성.

사람은 이 초안을 라벨링 도구에서 **교정만** 하면 되므로 라벨링이 빨라진다.
(초안이므로 오검출·오분류가 있을 수 있음 — 반드시 사람이 검수.)

사용:
    python -m src.joelchoi.realdomain.autolabel \
        --weights experiments/joelchoi/exp011_yolo11n_aug/weights/best.pt \
        --images data/realdomain_eval/images \
        --out    data/realdomain_eval/labels \
        --conf 0.25
"""

import argparse
from pathlib import Path


def autolabel(
    weights: str,
    images: str,
    out: str,
    conf: float = 0.25,
    iou: float = 0.45,
    imgsz: int = 640,
) -> None:
    from ultralytics import YOLO

    images = Path(images)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights))
    exts = ("*.jpg", "*.jpeg", "*.png")
    paths = sorted(p for e in exts for p in images.glob(e))
    print(f"오토라벨 대상: {len(paths)}장")

    # 모델 클래스 이름 → data.yaml 저장(라벨링 도구에서 약 이름 표시용)
    names = model.names if hasattr(model, "names") else {}
    import yaml

    with open(out.parent / "data.yaml", "w", encoding="utf-8") as f:
        yaml.dump({"names": names, "nc": len(names)}, f, allow_unicode=True)

    results = model.predict(
        source=[str(p) for p in paths],
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        stream=True,
        verbose=False,
    )
    manifest = []
    for p, res in zip(paths, results):
        lines = []
        b = res.boxes
        n = 0 if b is None else len(b)
        if n:
            # ultralytics는 정규화 xywh를 제공(res.boxes.xywhn)
            for cls, xywhn, cf in zip(
                b.cls.tolist(), b.xywhn.tolist(), b.conf.tolist()
            ):
                cx, cy, w, h = xywhn
                lines.append(f"{int(cls)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        (out / f"{p.stem}.txt").write_text("\n".join(lines), encoding="utf-8")
        manifest.append((p.name, n))

    total = sum(n for _, n in manifest)
    print(
        f"초안 라벨 저장: {out}  (총 박스 {total}개, 이미지당 평균 "
        f"{total / max(len(paths), 1):.1f})"
    )
    print("→ Label Studio / LabelImg에서 박스·클래스를 교정하세요.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--imgsz", type=int, default=640)
    a = ap.parse_args()
    autolabel(a.weights, a.images, a.out, a.conf, a.iou, a.imgsz)
