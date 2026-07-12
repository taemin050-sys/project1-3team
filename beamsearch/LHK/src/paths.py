"""HealthEat 경로 어댑터 (multi-machine 개인레포 내부, 자족형).
labkit(레포 루트)로 머신별 데이터 경로 자동 해석. 기존 실험 스크립트(import paths)가 그대로 동작.
코드·runs·SSOT는 이 레포 안(경로는 이 파일 기준 자동). 데이터·synth는 머신별 config로 참조."""

import sys
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[3]  # …/multi-machine  (레포 루트, labkit.py 위치)
PROJ = HERE.parents[1]  # …/projects/healtheat
sys.path.insert(0, str(REPO))
import labkit  # noqa: E402

_c = labkit.load("healtheat")  # 이 머신 경로 자동 해석(실패 시 친절한 에러로 중단)

MACHINE = labkit.MACHINE
DATA_ROOT = _c["data_root"]
PROCESSED = _c["processed"]  # synth 데이터셋 루트
TRAIN_IMAGES = DATA_ROOT / "train_images"
TRAIN_ANNOTATIONS = DATA_ROOT / "train_annotations"
TEST_IMAGES = DATA_ROOT / "test_images"

# 코드·runs·SSOT는 레포 로컬(이 파일 기준 → 어느 머신이든 자동으로 옳음)
LHK = PROJ  # (하위호환 별칭) 프로젝트 루트
RUNS = PROJ / "runs"  # 학습 산출물(로컬, gitignore)
SSOT = PROJ / "data" / "processed"  # class_map.json 등

# 공유 결과·요약은 labkit 버스로
RESULTS_DIR = _c["results_dir"]
LAB = _c["project_dir"]


def results_file():
    return labkit.results_file("healtheat")


if __name__ == "__main__":
    print(f"MACHINE   = {MACHINE}")
    print(f"REPO      = {REPO}")
    print(f"PROJ      = {PROJ}")
    print(f"DATA_ROOT = {DATA_ROOT}   (exists={DATA_ROOT.exists()})")
    print(
        f"  train/annot/test = {TRAIN_IMAGES.exists()}/{TRAIN_ANNOTATIONS.exists()}/{TEST_IMAGES.exists()}"
    )
    print(f"PROCESSED = {PROCESSED}   (exists={PROCESSED.exists()})")
    print(f"SSOT      = {SSOT}   (class_map={(SSOT / 'class_map.json').exists()})")
    print(f"RESULTS   = {results_file()}")
    print(
        "\n✅ 경로 확인 OK"
        if TRAIN_IMAGES.exists() and (SSOT / "class_map.json").exists()
        else "\n❌ 경로 문제"
    )
