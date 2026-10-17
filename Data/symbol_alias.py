"""
需要映射到美股代码的别名（用户不看韩股/港股，统一走美股上市代码）。

- SK海力士 在美股以 SKHY（NASDAQ，SK hynix Inc.）上市；KOSPI 代码 000660 在本项目内不使用，
  统一映射到 SKHY，避免被当成 A 股 6 位代码（SZ/SS）去验证或分析。
- 本模块只放"中文/别名 → 美股代码"的显式映射，不做模糊匹配。
  美股一般名称查询仍走 Data/us_stock_cik_map.csv。
"""

from __future__ import annotations

from typing import Optional

# 键为归一化后的写法（去空格、大写），见 _normalize
ALIAS_TO_US: dict[str, str] = {
    "海力士": "SKHY",
    "SK海力士": "SKHY",
    "SKHYNIX": "SKHY",
    "SKHY": "SKHY",
    "HYNIX": "SKHY",
    "000660": "SKHY",  # KOSPI 代码，统一映射到美股上市代码
}


def _normalize(text: str) -> str:
    return "".join((text or "").split()).upper()


def alias_symbol(text: str) -> Optional[str]:
    """若输入是已登记的别名，返回美股代码；否则返回 None。"""
    return ALIAS_TO_US.get(_normalize(text))
