"""
统一 LLM 工厂 — 所有 ChatOpenAI 实例的构造入口。

职责：
1. 从 Settings 读取 api_key / base_url / model_name，避免各处散落 os.getenv；
2. 针对 opencode zen / Console Go 端点自动注入 `x-opencode-session` header
   （官方要求：缺少该 header 会报 MissingSessionID / 无法高效路由）；
3. session 值来源：FINGENT_OPENCODE_SESSION 环境变量，缺省为进程级稳定 UUID
   （每个进程一个稳定会话即可满足 prompt 缓存优化要求）。
"""

import os
import uuid
from typing import Any, Dict, Optional

from langchain_openai import ChatOpenAI

from config import get_settings

# 进程级稳定会话 ID（OpenCode Go 要求每个会话一个稳定 ID 以便路由/缓存）
_PROCESS_SESSION = os.getenv("FINGENT_OPENCODE_SESSION") or str(uuid.uuid4())


def _session_headers(base_url: str, api_key: str) -> Optional[Dict[str, str]]:
    """opencode zen 端点自动附加 x-opencode-session；其他端点不加。"""
    if "opencode" in (base_url or "").lower():
        return {"x-opencode-session": _PROCESS_SESSION}
    return None


def create_chat_model(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model_name: Optional[str] = None,
    temperature: float = 0.5,
    **kwargs: Any,
) -> ChatOpenAI:
    """构造 ChatOpenAI。缺省参数取自 Settings（.env）。"""
    s = get_settings()
    api_key = api_key or s.qianwen_api_key
    base_url = base_url or s.model_base_url
    model_name = model_name or s.model_name

    headers = _session_headers(base_url, api_key)
    if headers:
        kwargs["default_headers"] = headers

    return ChatOpenAI(
        api_key=api_key,
        base_url=base_url,
        model=model_name,
        temperature=temperature,
        **kwargs,
    )