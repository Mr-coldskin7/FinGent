#!/usr/bin/env python3
"""演示前自检（只用标准库，不依赖虚拟环境）。

    python3 scripts/demo_preflight.py            # 检查环境、后端、模型配置、行情预热、测试（不消耗模型额度）
    python3 scripts/demo_preflight.py --ask      # 额外发一次真实问答，验证模型链路（消耗一次额度，约 5 秒）
    python3 scripts/demo_preflight.py --skip-tests

退出码：存在 ✕ 项时为 1，仅有 ! 警告时为 0。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT / ".env"
AUTH_TOKEN = ""  # 由 read_env() 填充；只用于请求头，永不打印

OK, WARN, FAIL = "✓", "!", "✕"
results: list[tuple[str, str, str]] = []


def record(level: str, name: str, detail: str = "") -> None:
    results.append((level, name, detail))
    print(f"  {level} {name}" + (f" — {detail}" if detail else ""))


def read_env() -> dict[str, str]:
    """解析 .env（只取键值，不打印值本身，避免密钥进入终端记录）。"""
    global AUTH_TOKEN
    env: dict[str, str] = {}
    if not ENV_FILE.exists():
        return env
    for raw in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        env[key.strip()] = value.strip().strip('"').strip("'")
    AUTH_TOKEN = env.get("FINGENT_API_TOKEN", "")
    return env


def http_json(url: str, timeout: float, method: str = "GET", body: dict | None = None):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    headers = {"Content-Type": "application/json"} if body is not None else {}
    if AUTH_TOKEN:
        headers["Authorization"] = f"Bearer {AUTH_TOKEN}"
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def check_env(env: dict[str, str]) -> None:
    print("\n[1] 环境配置（.env，只检查键是否存在）")
    if not ENV_FILE.exists():
        record(FAIL, ".env 不存在", "cp .env.example .env 并填写 QIANWEN_API_KEY")
        return
    for key in ("QIANWEN_API_KEY", "MODEL_BASE_URL", "MODEL_NAME"):
        if env.get(key):
            record(OK, f"{key} 已设置")
        else:
            record(FAIL, f"{key} 未设置")
    # run_backend.sh 不读取 .env，实际绑定地址由启动时的环境变量决定（默认 127.0.0.1）
    host = os.environ.get("API_HOST", "127.0.0.1")
    cors = env.get("CORS_ORIGINS", "*")
    token = env.get("FINGENT_API_TOKEN", "")
    if token:
        record(OK, f"访问令牌已启用（FINGENT_API_TOKEN，长度 {len(token)}，不显示内容）")
    elif host not in ("127.0.0.1", "localhost", "::1"):
        record(FAIL, f"监听 {host} 但未设置 FINGENT_API_TOKEN", "受保护接口会拒绝非本机访问；请设置令牌或改回 127.0.0.1")
    else:
        record(OK, f"未设置访问令牌，受保护接口仅本机可用（实际绑定 {host}）")
    if host == "0.0.0.0":
        record(WARN, "后端将监听 0.0.0.0（局域网可达）", "演示在会场网络下建议取消 API_HOST 环境变量")
    if cors == "*":
        record(WARN, "CORS_ORIGINS=*（任意网页可跨域调用后端）", "建议改为 http://localhost:5173")
    else:
        record(OK, "CORS 已限定来源")
    if env.get("BOCHA_API_KEY"):
        record(OK, "新闻检索已配置（BOCHA_API_KEY）")
    else:
        record(WARN, "新闻检索未启用（BOCHA_API_KEY 为空）",
               "舆情/基本面将缺少新闻背景；如需启用请在 .env 配置，或在 DEMO.md 中说明")
    if env.get("FINGENT_CHECKPOINTER", "memory") == "memory":
        record(OK, "Checkpointer = memory（无需 Redis）")


def check_backend(base: str) -> bool:
    print(f"\n[2] 后端服务（{base}）")
    try:
        http_json(f"{base}/health", timeout=3)
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        record(FAIL, "后端不可达", f"{exc}；请先运行 ./scripts/run_backend.sh")
        return False
    record(OK, "/health 正常")
    try:
        payload = http_json(f"{base}/api/v1/agents", timeout=5)
        items = payload.get("agents") or []
        names = [str(a.get("name")) for a in items if isinstance(a, dict) and a.get("name")]
        if len(names) >= 4:
            record(OK, f"Agent 注册表 {len(names)} 个", ", ".join(names))
        else:
            record(FAIL, "Agent 注册表异常", f"仅 {len(names)} 个")
    except Exception as exc:  # noqa: BLE001 - 仅用于诊断展示
        record(WARN, "无法读取 Agent 列表", str(exc))
    return True


def check_model(base: str) -> None:
    print("\n[3] 模型配置")
    try:
        cfg = http_json(f"{base}/api/v1/model/config", timeout=5).get("config", {})
        record(OK, f"模型 {cfg.get('model_name', '?')}", f"base_url={cfg.get('base_url', '?')}")
    except Exception as exc:  # noqa: BLE001
        record(FAIL, "读取模型配置失败", str(exc))


def check_market(base: str) -> None:
    print("\n[4] 行情预热（冷启动 A 股约 50 秒，此处会等待）")
    for market in ("zh", "us"):
        try:
            data = http_json(f"{base}/api/v1/market?market={market}&limit=20", timeout=120)
            if not data.get("success", False):
                record(WARN, f"行情 {market} 失败", str(data.get("error", "未知错误")))
                continue
            rows = data.get("stocks") or []
            source = data.get("source")
            age = data.get("age_s")
            detail = f"{len(rows)} 条"
            if source:
                detail += f"，source={source}"
            if age is not None:
                detail += f"，缓存年龄 {age}s"
            if rows:
                record(OK, f"行情 {market}", detail)
            else:
                record(WARN, f"行情 {market} 为空", "数据源可能限流，演示时行情页会显示提示")
        except Exception as exc:  # noqa: BLE001
            record(WARN, f"行情 {market} 失败", str(exc))


def check_ask(base: str) -> None:
    print("\n[5] 真实模型链路（--ask，消耗一次额度）")
    try:
        data = http_json(f"{base}/api/v1/chat", timeout=180, method="POST",
                         body={"user_input": "你好", "user_id": "preflight"})
        text = str(data.get("result") or data.get("response") or data)[:60]
        record(OK, "闲聊问答返回", text.replace("\n", " "))
    except Exception as exc:  # noqa: BLE001
        record(FAIL, "问答失败", f"{exc}（检查 key 余额 / 网络 / MODEL_BASE_URL）")


def check_frontend() -> None:
    print("\n[6] 前端")
    dist = ROOT / "Frontend" / "dist" / "index.html"
    if dist.exists():
        record(OK, "Frontend/dist 已构建")
    else:
        record(WARN, "Frontend/dist 未构建", "cd Frontend && npm run build（dev 模式可忽略）")
    try:
        urllib.request.urlopen("http://127.0.0.1:5173", timeout=2)
        record(OK, "开发服务器 :5173 在线")
    except Exception:  # noqa: BLE001
        record(WARN, "开发服务器 :5173 未启动", "cd Frontend && npm run dev")


def check_eval() -> None:
    print("\n[7] 评测中心数据")
    runs = sorted((ROOT / "Eval" / "runs").glob("*/report.json"))
    if runs:
        record(OK, f"已有 {len(runs)} 份评测报告", f"最新：{runs[-1].parent.name}")
    else:
        record(WARN, "没有评测报告", "uv run python -m Eval.reliability --k 5（约 25 分钟）")


def check_tests(skip: bool) -> None:
    print("\n[8] 测试套件")
    if skip:
        record(WARN, "已跳过（--skip-tests）")
        return
    proc = subprocess.run(
        ["uv", "run", "pytest", "Test", "-q", "-p", "no:cacheprovider"],
        cwd=ROOT, capture_output=True, text=True, timeout=300,
    )
    last = (proc.stdout.strip().splitlines() or ["(无输出)"])[-1]
    if proc.returncode == 0:
        record(OK, "pytest 通过", last)
    else:
        record(FAIL, "pytest 失败", last)


def main() -> int:
    parser = argparse.ArgumentParser(description="FinGent 演示前自检")
    parser.add_argument("--base", default=os.getenv("FINGENT_API_BASE", "http://127.0.0.1:8000"))
    parser.add_argument("--ask", action="store_true", help="发一次真实问答（消耗额度）")
    parser.add_argument("--skip-tests", action="store_true", help="跳过 pytest")
    args = parser.parse_args()
    base = args.base.rstrip("/")

    print("FinGent 演示前自检")
    env = read_env()
    check_env(env)
    if check_backend(base):
        check_model(base)
        check_market(base)
        if args.ask:
            check_ask(base)
    check_frontend()
    check_eval()
    check_tests(args.skip_tests)

    fails = [r for r in results if r[0] == FAIL]
    warns = [r for r in results if r[0] == WARN]
    print("\n" + "=" * 60)
    print(f"结果：{len(results) - len(fails) - len(warns)} 通过 · {len(warns)} 警告 · {len(fails)} 失败")
    if fails:
        print("存在失败项，请先处理后再演示。")
    elif warns:
        print("可以演示，但请留意警告项。")
    else:
        print("全部通过，可以演示。")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
