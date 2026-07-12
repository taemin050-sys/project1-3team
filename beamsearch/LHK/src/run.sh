#!/usr/bin/env bash
# 사용: bash scripts/run.sh <파이썬스크립트> <런이름>
# → 절전방지 + 영구로그(runs/logs) + 실시간 tee. 위치 독립(어느 머신이든 동작).
# python 경로는 LABKIT_PY 환경변수로 머신별 지정(기본=MacBook codeit env).
set -euo pipefail
PROJ="$(cd "$(dirname "$0")/.." && pwd)"
PY="${LABKIT_PY:-/Users/macbook/miniforge3/envs/codeit/bin/python}"
SCRIPT="$1"; NAME="${2:-run}"
mkdir -p "$PROJ/runs/logs"
LOG="$PROJ/runs/logs/${NAME}_$(date +%Y%m%d_%H%M%S).log"
echo "▶ 로그: $LOG"
echo "  실시간 보기:  tail -f \"$LOG\""
caffeinate -i env PYTORCH_ENABLE_MPS_FALLBACK=1 YOLO_CONFIG_DIR=/tmp/ultra_cfg \
  "$PY" -u "$SCRIPT" 2>&1 | tee "$LOG"
