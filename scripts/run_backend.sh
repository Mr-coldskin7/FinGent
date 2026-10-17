#!/bin/bash
# 后端守护进程：uvicorn 无论因何退出（包括原生层崩溃），2 秒内自动拉起。
# 用法：./scripts/run_backend.sh   （日志写入 logs/api.log）
cd "$(dirname "$0")/.." || exit 1
mkdir -p logs
while true; do
  echo "[$(date '+%F %T')] starting api_server" >> logs/api.log
  uv run uvicorn api_server:app --port 8000 >> logs/api.log 2>&1
  code=$?
  echo "[$(date '+%F %T')] api_server exited (code=$code), restarting in 2s" >> logs/api.log
  sleep 2
done
