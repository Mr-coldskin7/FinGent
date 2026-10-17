"""
板块/市场层面的实时背景（供"科技股还能买吗"这类不指定个股的问题使用）。

数据来自 Yahoo Finance 直连（见 yahoo_us.py），取一篮子代表性科技股的日涨跌。
只做背景参考，不是估值或买卖信号；失败时返回空字符串，由上层降级为通用回答。
"""

from __future__ import annotations

import time
from typing import Optional

from Data.providers.yahoo_us import fetch_us_quotes

# 代表性科技股篮子（覆盖平台、半导体、软件、云）
TECH_BASKET: list[str] = [
    "NVDA", "AAPL", "MSFT", "GOOGL", "META", "AMZN",
    "AVGO", "AMD", "TSM", "ORCL", "CRM", "ADBE", "INTC", "TSLA",
]

_TTL_S = 120
_cache: dict = {"ts": 0.0, "text": None, "stats": None}


def _format(quotes: list[dict]) -> tuple[str, dict]:
    up = sum(1 for q in quotes if q["changePercent"] > 0)
    down = sum(1 for q in quotes if q["changePercent"] < 0)
    avg = sum(q["changePercent"] for q in quotes) / len(quotes)
    ordered = sorted(quotes, key=lambda q: q["changePercent"], reverse=True)
    stats = {
        "count": len(quotes),
        "up": up,
        "down": down,
        "avg_change_percent": round(avg, 2),
        "top": [q["symbol"] for q in ordered[:3]],
        "bottom": [q["symbol"] for q in ordered[-3:]],
    }
    lines = [
        f"- {q['symbol']}（{q['name']}）：现价 {q['price']}，日涨跌 {q['changePercent']:+.2f}%"
        for q in ordered
    ]
    header = (
        f"美股科技篮子（{len(quotes)} 只）：上涨 {up} 只，下跌 {down} 只，"
        f"平均日涨跌 {avg:+.2f}%。"
    )
    return header + "\n" + "\n".join(lines), stats


async def tech_market_context() -> tuple[str, Optional[dict]]:
    """返回 (可直接放进提示词的文本, 统计信息)。取不到数据时返回 ("", None)。"""
    now = time.time()
    if _cache["text"] is not None and now - _cache["ts"] < _TTL_S:
        return _cache["text"], _cache["stats"]
    try:
        quotes = await fetch_us_quotes(TECH_BASKET)
    except Exception:
        quotes = []
    if len(quotes) < max(3, len(TECH_BASKET) // 2):
        # 数据不足时不缓存，下次再试；返回空，让上层不编造数字
        return "", None
    text, stats = _format(quotes)
    _cache.update(ts=now, text=text, stats=stats)
    return text, stats
