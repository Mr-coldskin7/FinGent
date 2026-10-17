#!/bin/bash
# 后端守护进程：uvicorn 无论因何退出（包括原生层崩溃），2 秒内自动拉起。
# 用法：./scripts/run_backend.sh   （日志写入 logs/api.log）
cd "$(dirname "$0")/.." || exit 1
mkdir -p logs
# 尊重 .env 的 API_HOST/API_PORT（review P3）；二次重启退避，避免绑定失败时刷爆日志
API_HOST="${API_HOST:-127.0.0.1}"  # 默认仅本机可达；需要局域网访问时显式设置 API_HOST=0.0.0.0
API_PORT="${API_PORT:-8000}"
backoff=2
while true; do
  echo "[$(date '+%F %T')] starting api_server" >> logs/api.log
  uv run uvicorn api_server:app --host "$API_HOST" --port "$API_PORT" >> logs/api.log 2>&1
  code=$?
  if [ "$code" -eq 0 ]; then
    backoff=2
  else
    backoff=$(( backoff < 30 ? backoff * 2 : 30 ))
  fi
  echo "[$(date '+%F %T')] api_server exited (code=$code), restarting in ${backoff}s" >> logs/api.log
  sleep "$backoff"
done
