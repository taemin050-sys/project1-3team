"""lab-kit — 멀티머신 분산작업 공용 툴킷 (프로젝트 무관, 재사용).

핵심 원칙: **git을 버스로.** 코드·결과(작음)만 git 공유, 데이터·weights(무거움)는 각 머신 로컬.
세션은 머신 간 공유 불가 → 각 머신이 자기 세션을 돌고, 결과는 머신별 파일로 이 repo에 커밋.

사용 (프로젝트 스크립트에서):
    import os, sys
    sys.path.insert(0, os.path.expanduser(os.environ.get("LABKIT_HOME", "~/lab-kit")))
    import labkit
    cfg = labkit.load("<project>")          # 이 머신의 경로 자동 해석
    labkit.record("<project>", {...})       # 결과 1건 append (머신별 파일)
    done = labkit.done_keys("<project>", ["variant", "fold"])  # skip-done용

프로젝트 등록: projects/<name>/config.json 에 roots(머신별 후보 경로) 선언.
"""

import glob
import json
import os
import re
import socket
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECTS = ROOT / "projects"

# 머신 식별 (env override 우선 → 실수 방지)
MACHINE = (
    re.sub(
        r"[^A-Za-z0-9]+",
        "-",
        os.environ.get("LAB_MACHINE") or socket.gethostname().split(".")[0],
    )
    .strip("-")
    .lower()
    or "unknown"
)


def _results_dir(project):
    d = PROJECTS / project / "results"
    d.mkdir(parents=True, exist_ok=True)
    return d


def load(project):
    """projects/<project>/config.json 의 roots를 이 머신에서 해석해 dict로 반환.
    각 root는 후보 경로 리스트 — 첫 존재 경로 채택. 못 찾으면 친절한 에러로 즉시 중단."""
    pdir = PROJECTS / project
    cfg_path = pdir / "config.json"
    assert cfg_path.exists(), (
        f"[labkit] 프로젝트 '{project}' 미등록 — {cfg_path} 없음. templates/config.json 참고해 생성."
    )
    cfg = json.load(open(cfg_path))
    resolved = {
        "machine": MACHINE,
        "project": project,
        "results_dir": _results_dir(project),
        "project_dir": pdir,
    }
    for name, candidates in cfg.get("roots", {}).items():
        hit = next(
            (
                Path(os.path.expanduser(c))
                for c in candidates
                if Path(os.path.expanduser(c)).exists()
            ),
            None,
        )
        assert hit is not None, (
            f"[labkit] project='{project}' root='{name}' 를 이 머신({MACHINE})에서 못 찾음.\n"
            f"  → projects/{project}/config.json 의 roots.{name} 후보에 이 머신 경로를 추가하세요.\n"
            f"  현재 후보: {candidates}"
        )
        resolved[name] = hit
    return resolved


def results_file(project):
    """이 머신 전용 결과 파일 (병합충돌 0)."""
    return _results_dir(project) / f"{MACHINE}.jsonl"


def record(project, rec):
    """결과 1건 append (머신 태그 자동)."""
    rec = {**rec, "machine": MACHINE}
    with open(results_file(project), "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_records(project):
    """모든 머신의 results/*.jsonl 병합 로드 (_src=파일명 태그)."""
    out = []
    for f in sorted(glob.glob(str(_results_dir(project) / "*.jsonl"))):
        for ln in open(f):
            if ln.strip():
                r = json.loads(ln)
                r["_src"] = os.path.basename(f)
                out.append(r)
    return out


def done_keys(project, keys):
    """이미 끝난 레코드의 (keys...) 튜플 집합 — skip-done용."""
    return {
        tuple(r[k] for k in keys)
        for r in load_records(project)
        if all(k in r for k in keys)
    }


if __name__ == "__main__":
    import sys

    print(f"MACHINE = {MACHINE}")
    print(f"ROOT    = {ROOT}")
    print(
        f"projects: {[p.name for p in PROJECTS.iterdir() if p.is_dir()] if PROJECTS.exists() else []}"
    )
    if len(sys.argv) > 1:
        print(
            json.dumps(
                {k: str(v) for k, v in load(sys.argv[1]).items()},
                ensure_ascii=False,
                indent=1,
            )
        )
