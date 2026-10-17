"""
AKShare 调用的进程隔离层。

背景：akshare 部分接口（如 stock_zh_a_spot）内部使用 py_mini_racer（V8）解析 JS。
在 API 进程的线程池里并发调用时，V8 可能触发原生层 FATAL（abort），
整个 uvicorn 进程随之退出，前端表现为"network error"。

做法：所有 akshare 调用放到独立的子进程池执行。子进程崩溃只影响该次调用，
主进程捕获后重建进程池并返回可读错误，API 服务本身保持在线。
"""

from __future__ import annotations

import asyncio
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from typing import Any

_executor: ProcessPoolExecutor | None = None


def _worker(func_name: str, kwargs: dict) -> Any:
    """子进程内执行：在子进程里导入 akshare 并调用指定函数。"""
    import akshare as ak

    return getattr(ak, func_name)(**kwargs)


def _get_executor() -> ProcessPoolExecutor:
    global _executor
    if _executor is None:
        # spawn：避免 fork 继承主进程里已初始化的 V8/网络状态
        _executor = ProcessPoolExecutor(
            max_workers=2, mp_context=mp.get_context("spawn")
        )
    return _executor


def _reset_executor() -> None:
    global _executor
    if _executor is not None:
        _executor.shutdown(wait=False, cancel_futures=True)
    _executor = None


async def ak_call(func_name: str, *, timeout: float = 120, **kwargs: Any) -> Any:
    """在隔离子进程中调用 akshare.<func_name>(**kwargs)，返回结果（DataFrame 等可 pickle 对象）。

    - 进程池已损坏（上次子进程崩溃）：自动重建并重试一次。
    - 仍失败：抛出 RuntimeError（中文，可直接展示给用户），绝不让异常穿透成 500 / 连接断开。
    """
    loop = asyncio.get_running_loop()
    last_error: Exception | None = None
    for attempt in (1, 2):
        try:
            fut = loop.run_in_executor(_get_executor(), _worker, func_name, kwargs)
            return await asyncio.wait_for(fut, timeout=timeout)
        except BrokenProcessPool as e:
            last_error = e
            _reset_executor()  # 下一轮会新建进程池
            continue
        except asyncio.TimeoutError as e:
            raise RuntimeError(f"行情数据获取超时（{func_name}），请稍后重试") from e
    raise RuntimeError(f"行情数据服务暂时不可用（{func_name}），请稍后重试") from last_error
