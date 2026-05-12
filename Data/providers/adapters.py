"""
Adapters - 数据转换层
将原始数据转换为标准 Unified Models
"""

from typing import List, Optional
import sys
import os

import pandas as pd

# 导入模型
from Data.models import BasicInfo, PriceData, CompanyDescription, FinancialReport


def adapt_us_basic(raw: list, symbol: str) -> Optional[BasicInfo]:
    """美股基础信息转换"""
    if not raw or not isinstance(raw, list):
        return None
    item = raw[0]
    return BasicInfo(
        symbol=item.get("symbol", symbol),
        name=item.get("name", ""),
        currency=item.get("currency", "USD"),
        exchange=item.get("exchange", ""),
        exchange_full_name=item.get("exchangeFullName", ""),
        market_type="US",
    )


def adapt_us_prices(raw: list, symbol: str) -> List[PriceData]:
    """美股价格转换"""
    if not raw:
        return []
    return [
        PriceData(
            symbol=symbol,
            date=item.get("date", ""),
            open=float(item.get("open", 0)),
            high=float(item.get("high", 0)),
            low=float(item.get("low", 0)),
            close=float(item.get("close", 0)),
            volume=int(item.get("volume", 0)),
            adjusted_close=float(item.get("adjClose", item.get("close", 0))),
            market_type="US",
        )
        for item in raw
    ]


def adapt_us_description(raw: dict, symbol: str) -> Optional[CompanyDescription]:
    """美股公司描述转换"""
    if not raw:
        return None
    return CompanyDescription(
        symbol=symbol,
        description=raw.get("description", ""),
        ceo=raw.get("CEO", ""),
        headquarters=f"{raw.get('hq_address1', '')} {raw.get('hq_city', '')}",
        founded=raw.get("original_filing_date", ""),
        market_type="US",
    )


def adapt_us_reports(k_urls: list, q_urls: list, symbol: str) -> List[FinancialReport]:
    """美股财报转换"""
    reports = []
    for url in k_urls or []:
        reports.append(
            FinancialReport(
                symbol=symbol,
                report_type="10-K",
                filing_date="",
                url=url,
                title=f"10-K for {symbol}",
                market_type="US",
            )
        )
    for url in q_urls or []:
        reports.append(
            FinancialReport(
                symbol=symbol,
                report_type="10-Q",
                filing_date="",
                url=url,
                title=f"10-Q for {symbol}",
                market_type="US",
            )
        )
    return reports


def adapt_cn_basic(raw, symbol: str) -> Optional[BasicInfo]:
    """A股基础信息转换"""
    if raw is None or raw.empty:
        return None

    # akshare返回的是DataFrame，格式为：
    #        item        value
    # 0    股票代码      600519
    # 1    股票简称      贵州茅台
    # 需要把item列作为key，value列作为value
    if "item" in raw.columns and "value" in raw.columns:
        # 转换为字典：{'股票代码': '600519', '股票简称': '贵州茅台', ...}
        info_dict = dict(zip(raw["item"], raw["value"]))
        name = info_dict.get("股票简称", "") or info_dict.get("公司名称", "")
        exchange = info_dict.get("交易所", "")
    else:
        # 备用：如果格式变了，尝试原来的方式
        row = raw.iloc[0]
        name = row.get("公司名称", "") if "公司名称" in raw.columns else ""
        exchange = row.get("交易所", "") if "交易所" in raw.columns else ""

    return BasicInfo(
        symbol=symbol,
        name=name,
        currency="CNY",
        exchange=exchange,
        exchange_full_name="",
        market_type="CN",
    )


def adapt_cn_prices(df, symbol: str) -> List[PriceData]:
    """A股价格转换（新浪财经数据源）"""
    if df is None or df.empty:
        return []
    return [
        PriceData(
            symbol=symbol,
            date=str(row.get("date", ""))[:10],
            open=float(row.get("open", 0)) if pd.notna(row.get("open")) else 0,
            high=float(row.get("high", 0)) if pd.notna(row.get("high")) else 0,
            low=float(row.get("low", 0)) if pd.notna(row.get("low")) else 0,
            close=float(row.get("close", 0)) if pd.notna(row.get("close")) else 0,
            volume=int(row.get("volume", 0)) if pd.notna(row.get("volume")) else 0,
            market_type="CN",
        )
        for _, row in df.iterrows()
    ]


def adapt_cn_financial(raw, symbol: str) -> List[FinancialReport]:
    """A股财报转换（简化版）"""
    # A股财报格式较复杂，这里做简化处理
    return [
        FinancialReport(
            symbol=symbol,
            report_type="Annual",
            filing_date="",
            url="",
            title=f"Financial Report for {symbol}",
            market_type="CN",
        )
    ]
