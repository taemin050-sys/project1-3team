#!/usr/bin/env bash
# 사용: bash scripts/run.sh <파이썬스크립트> <런이름>
# → 절전방지 + 영구로그(runs/logs) + 실시간 tee. 예: bash scripts/run.sh scripts/tv_baselines.py tv
set -euo pipefail
LHK="/Users/macbook/dev/learning/codeit/01_Proj_HealthEat_cla/project1-3team/beamsearch/LHK"
PY="/Users/macbook/miniforge3/envs/codeit/bin/python"
SCRIPT="$1"; NAME="${2:-run}"
mkdir -p "$LHK/runs/logs"
LOG="$LHK/runs/logs/${NAME}_$(date +%Y%m%d_%H%M%S).log"
echo "▶ 로그: $LOG"
echo "  실시간 보기:  tail -f \"$LOG\""
caffeinate -i env PYTORCH_ENABLE_MPS_FALLBACK=1 YOLO_CONFIG_DIR=/tmp/ultra_cfg \
  "$PY" -u "$SCRIPT" 2>&1 | tee "$LOG"
