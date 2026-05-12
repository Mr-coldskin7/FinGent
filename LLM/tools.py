import os, datetime as dt, pandas as pd
import sys, os
from typing import Optional

ROOT = os.path.normpath(os.path.join(__file__, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from datetime import timedelta
from langchain.tools import tool
from pydantic import BaseModel, Field
from Data.providers import us_stock, zh_stock
from langchain.tools import tool


class Symbol_Args(BaseModel):
    symbol: str = Field("AAPL", description="美股代码，如 AAPL")


class Name_Args(BaseModel):
    name: str = Field("AAPL", description="美股名称，如 Apple")


class Daily_Args(BaseModel):
    symbol: str = Field("AAPL", description="美股代码，如 AAPL")
    startDate: Optional[str] = Field(
        None, description="开始日期，YYYY-MM-DD，默认三个月前"
    )
    endDate: Optional[str] = Field(None, description="结束日期，YYYY-MM-DD，默认今天")
    resampleFreq: str = Field(
        "daily", description="重采样频率：daily/weekly/monthly/annually，默认 daily"
    )


class Tag_Args(BaseModel):
    tag: str = Field("election", description="标签，如 election")


class Financial_Report_Args(BaseModel):
    symbol: str = Field("AAPL", description="美股代码，如 AAPL")
    num: int = Field(1, description="报告数量，默认 1")


class zh_stock_symbol(BaseModel):
    symbol: str = Field("贵州茅台", description="股票代码如“600519”,或是股票名称")


@tool(args_schema=Daily_Args)
async def get_us_stock_price(
    symbol: str,
    startDate: Optional[str] = None,
    endDate: Optional[str] = None,
    resampleFreq: str = "daily",
) -> str:
    """
    Get the latest historical stock price data for a US stock by ticker symbol.
    Returns daily OHLCV data for the past 5 trading days.
    Example input: 'AAPL'
    """
    result = await us_stock.get_historical_stock_price_by_symbol(
        symbol, startDate, endDate, resampleFreq
    )
    return str(result)


@tool(args_schema=Symbol_Args)
async def get_us_stock_info_by_symbol(symbol: str) -> str:
    """
    get company basic info by symbol
    """
    result = await us_stock.get_basic_info_by_symbol(symbol)
    return str(result)


@tool(args_schema=Symbol_Args)
async def get_description_of_us_stock(symbol: str) -> str:
    """
    get company description by symbol
    """
    result = await us_stock.get_description(symbol)
    return str(result)


# @tool(args_schema=Symbol_Args)
# async def get_relative_news_by_stock(symbol: str) -> str:
#     """
#     get recent news by stock symbol
#     """
#     result = await us_stock.get_recent_news(symbol)
#     return str(result)

# @tool(args_schema=Tag_Args)
# async def get_relative_news_by_tag(tag: str) -> str:
#     """
#     get recent news by tag
#     """
#     result = await us_stock.get_relative_news_by_tag(tag)
#     return str(result)


@tool(args_schema=Symbol_Args)
async def fuzzy_search_us_symbols(symbol: str) -> str:
    """
    fuzzy search us stock symbols
    !!! It does not work in Chinese !!!
    """
    result = await us_stock.fuzzy_search_symbols(symbol)
    return str(result)


@tool(args_schema=Financial_Report_Args)
async def get_10K_financial_report(symbol: str, num: int = 1) -> str:
    """
    get company's 10k financial report by symbol
    """
    result = await us_stock.get_10K_financial_report(symbol, num)
    return str(result)


@tool(args_schema=Financial_Report_Args)
async def get_10Q_financial_report(symbol: str, num: int = 1) -> str:
    """
    get company's 10q financial report by symbol
    """
    result = await us_stock.get_10Q_financial_report(symbol, num)
    return str(result)


@tool(args_schema=Symbol_Args)
async def get_extracted_10Q_financial_report_by_symbol(symbol: str) -> str:
    """
    get extracted company's 10q financial report by symbol
    """
    result = await us_stock.get_financial_report(symbol)
    return str(result)


@tool(args_schema=zh_stock_symbol)
async def get_zh_stock_info(symbol: str) -> str:
    """
    get zh stock info by symbol or name
    """
    result = await zh_stock.stock_individual_info(symbol)
    return str(result)


@tool
async def get_see_summary() -> str:
    """
    get see summary
    """
    result = await zh_stock.get_see_summary()
    return str(result)


@tool
async def get_sz_stock_info() -> str:
    """
    get sz stock info
    """
    result = await zh_stock.get_sz_stock_info()
    return str(result)


@tool(args_schema=zh_stock_symbol)
async def stock_info_sh_name_code(symbol: str) -> str:
    """
    param symbol: choice of {"主板A股", "主板B股", "科创板"}
    获得主板信息，不是个股信息
    """
    result = await zh_stock.stock_info_sh_name_code(symbol)
    return str(result)


@tool(args_schema=zh_stock_symbol)
async def get_zh_stock_individual_price_info_recent(symbol: str) -> str:
    """
    get stock individual price info recent
    """
    result = await zh_stock.stock_individual_price_info_recent(symbol)
    return str(result)


@tool()
async def recent_zh_stock_list() -> str:
    """
    get recent stock list
    """
    result = await zh_stock.recent_stock_list()
    return str(result)


@tool(args_schema=zh_stock_symbol)
async def get_zh_financial_report(symbol: str) -> str:
    """
    get zh stock financial report by symbol or name
    """
    result = await zh_stock.get_financial_report(symbol)
    return str(result)


@tool(args_schema=zh_stock_symbol)
async def get_zh_stock_price(symbol: str) -> str:
    """
    get zh stock price by code or name
    """
    result = await zh_stock.get_historical_stock_price_by_symbol(symbol)
    return str(result)


@tool(args_schema=zh_stock_symbol)
async def get_zh_stock_debt(symbol: str) -> str:
    """
    get zh stock debt by code or name
    """
    result = await zh_stock.get_debt(symbol)
    return str(result)
