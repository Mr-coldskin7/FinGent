"""
Checkpointer 工厂 — memory | redis 可插拔切换

用法（同步场景，如 CLI / 模块加载）:
    checkpointer, cleanup = get_checkpointer()
    ...
    cleanup()

用法（FastAPI 请求内创建临时 graph 的场景）:
    checkpointer, cleanup = await get_checkpointer_async()
    try:
        ...
    finally:
        if cleanup:
            await cleanup()

环境变量:
    FINGENT_CHECKPOINTER=memory (默认) | redis
    REDIS_URL=redis://localhost:6379/0   (redis 模式)
"""

import asyncio
import os


def _mode() -> str:
    return os.getenv("FINGENT_CHECKPOINTER", "memory").strip().lower() or "memory"


def _redis_ctx():
    from langgraph.checkpoint.redis.aio import AsyncRedisSaver

    from config import get_settings

    return AsyncRedisSaver.from_conn_string(get_settings().redis_url)


def get_checkpointer():
    """同步入口（模块加载 / CLI 场景）。

    Returns:
        (checkpointer, cleanup_sync) — cleanup 为可选的同步回收函数。
    """
    mode = _mode()
    if mode == "redis":
        ctx = _redis_ctx()
        checkpointer = asyncio.run(ctx.__aenter__())

        def cleanup():
            try:
                asyncio.run(ctx.__aexit__(None, None, None))
            except Exception:
                pass

        return checkpointer, cleanup

    from langgraph.checkpoint.memory import MemorySaver

    return MemorySaver(), lambda: None


async def get_checkpointer_async():
    """异步入口（FastAPI 请求内创建临时 graph 的场景）。

    Returns:
        (checkpointer, cleanup_async | None) — memory 模式无需回收。
    """
    mode = _mode()
    if mode == "redis":
        ctx = _redis_ctx()
        checkpointer = await ctx.__aenter__()

        async def cleanup(*args):
            await ctx.__aexit__(*args)

        return checkpointer, cleanup

    from langgraph.checkpoint.memory import MemorySaver

    return MemorySaver(), None