#!/usr/bin/env bash
# 현재 학습 상황 한눈에: 프로세스 · 최신 로그 tail · baselines 표
LHK="/Users/macbook/dev/learning/codeit/01_Proj_HealthEat_cla/project1-3team/beamsearch/LHK"
PY="/Users/macbook/miniforge3/envs/codeit/bin/python"
echo "=== 실행 중 학습 프로세스 (경과·CPU) ==="
ps -Ao etime,%cpu,command | grep -E "scripts/.*\.py" | grep -v grep || echo "(없음)"
echo; echo "=== 최신 로그 마지막 10줄 ==="
LOG=$(ls -t "$LHK/runs/logs/"*.log 2>/dev/null | head -1 || true)
if [ -n "${LOG:-}" ]; then echo "$LOG"; tail -10 "$LOG"; else echo "(로그 없음)"; fi
echo; echo "=== baselines.json ==="
"$PY" - <<PYEOF
import json,os
f="$LHK/runs/baselines.json"
if os.path.exists(f):
    for x in sorted(json.load(open(f)),key=lambda a:-a['mAP_75_95']): print(f"  {x['model']:<18}{x['mAP_75_95']}")
else: print("  없음")
PYEOF
