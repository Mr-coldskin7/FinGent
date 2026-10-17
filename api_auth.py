"""API 访问控制（鉴权）。

判定规则（默认拒绝，所有决策集中在本文件，便于审计）：

1. 非 /api/v1 路径（/health、/docs 等）不做鉴权。
2. OPTIONS 预检请求放行：CORS 预检不携带凭证，本身不授予任何权限，真正的请求仍会被校验。
3. /api/v1 下，PUBLIC_PATH_PATTERNS 之外的接口全部需要鉴权：
   - 配置了 FINGENT_API_TOKEN：必须带 `Authorization: Bearer <token>`，否则 401。
   - 未配置 token：只放行回环地址（127.0.0.1 / ::1 / localhost），其它来源 403。
     这是单机开发与演示的安全默认值：不用额外配置，但局域网内其它机器打不进来。

token 比较使用 hmac.compare_digest（常量时间），不通过 query 参数传递（避免进入访问日志）。
"""

from __future__ import annotations

import hmac
import re
from dataclasses import dataclass
from typing import Optional

MIN_TOKEN_LENGTH = 16
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})
API_PREFIX = "/api/v1/"

# 公开只读接口：不含用户数据，也不产生模型调用成本。
PUBLIC_PATH_PATTERNS = (
    re.compile(r"^/api/v1/market$"),
    re.compile(r"^/api/v1/eval/runs(/.*)?$"),
    re.compile(r"^/api/v1/backtest-chart$"),
)


@dataclass(frozen=True)
class AuthDecision:
    allowed: bool
    status: int = 200
    code: str = "ok"
    message: str = ""


def validate_token(token: str) -> None:
    """启动时校验 token：空串表示未启用；启用则长度必须足够。"""
    if token and len(token) < MIN_TOKEN_LENGTH:
        raise ValueError(
            f"FINGENT_API_TOKEN 长度至少 {MIN_TOKEN_LENGTH} 位（当前 {len(token)} 位）。"
            "生成方式：python3 -c \"import secrets; print(secrets.token_urlsafe(32))\""
        )


def is_public(method: str, path: str) -> bool:
    if method.upper() == "OPTIONS":
        return True
    if not path.startswith(API_PREFIX):
        return True
    return any(p.match(path) for p in PUBLIC_PATH_PATTERNS)


def _bearer_value(authorization: Optional[str]) -> str:
    scheme, _, value = (authorization or "").strip().partition(" ")
    return value.strip() if scheme.lower() == "bearer" else ""


def decide(
    *,
    method: str,
    path: str,
    authorization: Optional[str],
    client_host: str,
    token: str,
) -> AuthDecision:
    """纯函数：给定请求要素，返回是否放行。不依赖框架对象，便于单测。"""
    if is_public(method, path):
        return AuthDecision(True)

    if token:
        presented = _bearer_value(authorization)
        if presented and hmac.compare_digest(presented.encode("utf-8"), token.encode("utf-8")):
            return AuthDecision(True)
        return AuthDecision(
            False, 401, "unauthorized",
            "需要有效的 API 访问令牌：请在「系统设置 → 访问令牌」中填写",
        )

    if client_host in LOOPBACK_HOSTS:
        return AuthDecision(True)
    return AuthDecision(
        False, 403, "forbidden",
        "服务未配置 API 访问令牌，仅允许本机访问（设置 FINGENT_API_TOKEN 后可远程访问）",
    )
