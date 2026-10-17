#!/bin/bash
# 决策结果前向验证 + 客观权重更新（供 launchd/cron 定时调用）。
# 用法：./scripts/run_outcome.sh [--no-weights]
#   --no-weights：只验证结果，不更新客观权重（权重更新受 MIN_VERIFIED_SAMPLES=30 门槛保护）
cd "$(dirname "$0")/.." || exit 1
mkdir -p logs
log=logs/outcome.log
echo "[$(date '+%F %T')] start Data.outcome $*" >> "$log"
uv run python -m Data.outcome "$@" >> "$log" 2>&1
code=$?
echo "[$(date '+%F %T')] Data.outcome exited (code=$code)" >> "$log"
exit "$code"
