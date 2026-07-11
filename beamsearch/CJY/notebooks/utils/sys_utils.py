import sys
from pathlib import Path

# CJY 프로젝트 루트를 식별하는 마커: 이 두 폴더가 함께 있는 디렉토리가 루트다.
# (monorepo 상위 폴더(beamsearch, project1_3team)에는 이 조합이 없어 오탐이 없다.)
_ROOT_MARKERS = ("src", "configs")


def _find_project_root(start: Path) -> Path | None:
    """start부터 위로 올라가며 _ROOT_MARKERS를 모두 가진 디렉토리를 찾는다."""
    for candidate in [start, *start.parents]:
        if all((candidate / marker).is_dir() for marker in _ROOT_MARKERS):
            return candidate
    return None


def add_experiment_root(parent_level: int | None = None) -> Path:
    """호출한 파일 기준으로 프로젝트 루트를 찾아 sys.path에 추가합니다.

    주피터 노트북 환경과 일반 .py 실행 환경을 모두 지원합니다.

    기본 동작(parent_level 생략 시)은 ``_ROOT_MARKERS``(src/, configs/)를 가진
    가장 가까운 상위 디렉토리를 자동으로 프로젝트 루트로 인식합니다. 노트북의
    커널 작업 디렉토리(cwd)가 실행 환경마다 달라도(예: notebooks/ vs 워크스페이스
    루트) 항상 동일한 루트를 가리키므로, 이전처럼 ``parent_level`` 숫자에 의존해
    노트북마다 다른 경로가 계산되는 문제가 없습니다.

    Args:
        parent_level (int | None, optional): 레거시 호환용. 지정하면 이전 방식대로
            현재 경로에서 정확히 이 레벨만큼 위로 이동합니다(자동 감지 미사용).
            새로 작성하는 코드는 인자 없이 호출하는 것을 권장합니다.

    Returns:
        Path: sys.path에 추가된 프로젝트 루트의 Path 객체.
    """
    # 주피터 노트북 환경과 일반 .py 환경을 모두 지원
    try:
        current_path = Path(__file__).resolve().parent
    except NameError:
        # 주피터 노트북 환경일 경우 현재 작업 디렉토리 기준
        current_path = Path(".").resolve()

    if parent_level is not None:
        # 레거시 모드: 지정한 레벨만큼 위로 이동(.parents는 0부터 시작하므로 인덱스 조절)
        project_root = current_path.parents[parent_level - 1]
    else:
        project_root = _find_project_root(current_path)
        if project_root is None:
            # 마커를 못 찾으면 기존 기본값(부모 1단계)으로 안전하게 폴백
            project_root = (
                current_path.parents[0] if current_path.parents else current_path
            )

    root_str = str(project_root)
    if root_str not in sys.path:
        sys.path.append(root_str)  # insert(0)보다 append가 부작용이 적어 안전합니다.

    return project_root
