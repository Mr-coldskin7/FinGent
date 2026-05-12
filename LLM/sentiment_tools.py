"""
Sentiment Analysis Tools - 情感分析工具集
基于关键词匹配和Web搜索的情感分析工具，可后续替换为专用情感模型
"""

import asyncio
import json
import sys
import os
import re
from typing import List, Optional

# 添加项目根目录到 Python 路径（确保能导入 Data 模块）
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from langchain.tools import tool
from pydantic import BaseModel, Field

from Data.providers.web_search import search as web_search


# ==================== 情感词典（可替换为模型推理）====================

POSITIVE_KEYWORDS = [
    # 中文
    "上涨",
    "利好",
    "突破",
    "增长",
    "盈利",
    "大涨",
    "飙升",
    "强劲",
    "乐观",
    "复苏",
    "回暖",
    "创新高",
    "超预期",
    "增持",
    "买入",
    "推荐",
    "看好",
    # 英文
    "beat",
    "surge",
    "rally",
    "growth",
    "profit",
    "gain",
    "rise",
    "soar",
    "bullish",
    "outperform",
    "upgrade",
    "strong",
    "recovery",
    "boom",
    "positive",
    "optimistic",
    "momentum",
    "breakthrough",
    "exceed",
]

NEGATIVE_KEYWORDS = [
    # 中文
    "下跌",
    "利空",
    "暴跌",
    "亏损",
    "裁员",
    "大跌",
    "疲软",
    "悲观",
    "衰退",
    "下滑",
    "回落",
    "创新低",
    "不及预期",
    "减持",
    "卖出",
    "看空",
    "预警",
    # 英文
    "drop",
    "crash",
    "loss",
    "layoff",
    "decline",
    "fall",
    "plunge",
    "tumble",
    "bearish",
    "underperform",
    "downgrade",
    "weak",
    "recession",
    "slump",
    "negative",
    "pessimistic",
    "miss",
    "warning",
    "risk",
    "concern",
]


def _keyword_sentiment_score(texts: List[str]) -> dict:
    """
    基于关键词匹配计算情感得分（fallback 方案）
    后续可替换为调用情感分析模型
    """
    positive_count = 0
    negative_count = 0
    neutral_count = 0
    text_scores = []

    for text in texts:
        if not text:
            neutral_count += 1
            text_scores.append({"text": text, "sentiment": "neutral", "score": 0})
            continue

        text_lower = text.lower()
        pos_hits = sum(1 for kw in POSITIVE_KEYWORDS if kw in text_lower)
        neg_hits = sum(1 for kw in NEGATIVE_KEYWORDS if kw in text_lower)

        if pos_hits > neg_hits:
            sentiment = "positive"
            score = min(pos_hits - neg_hits, 5)
            positive_count += 1
        elif neg_hits > pos_hits:
            sentiment = "negative"
            score = -min(neg_hits - pos_hits, 5)
            negative_count += 1
        else:
            sentiment = "neutral"
            score = 0
            neutral_count += 1

        text_scores.append(
            {
                "text": text[:200] + "..." if len(text) > 200 else text,
                "sentiment": sentiment,
                "score": score,
                "positive_hits": pos_hits,
                "negative_hits": neg_hits,
            }
        )

    total = len(texts) if texts else 1
    overall_score = (positive_count - negative_count) / total

    if overall_score > 0.2:
        overall = "positive"
    elif overall_score < -0.2:
        overall = "negative"
    else:
        overall = "neutral"

    return {
        "overall_sentiment": overall,
        "overall_score": round(overall_score, 4),
        "counts": {
            "positive": positive_count,
            "negative": negative_count,
            "neutral": neutral_count,
            "total": len(texts),
        },
        "details": text_scores,
        "method": "keyword_matching",  # 标记方法，便于后续替换
    }


def _extract_news_items(search_result: dict) -> List[dict]:
    """从 bocha API 返回结果中提取新闻条目"""
    items = []
    if not search_result or not isinstance(search_result, dict):
        return items

    # bocha API 数据结构: data -> webPages -> value[]
    data = search_result.get("data", {})
    web_pages = data.get("webPages", {}) if isinstance(data, dict) else {}
    values = web_pages.get("value", []) if isinstance(web_pages, dict) else []

    for item in values:
        if not isinstance(item, dict):
            continue
        items.append(
            {
                "title": item.get("name", ""),
                "summary": item.get("snippet", ""),
                "url": item.get("url", ""),
                "source": item.get("siteName", ""),
                "published_date": item.get("dateLastCrawled", ""),
            }
        )

    return items


def _is_chinese_stock(symbol: str) -> bool:
    """简单判断是否为 A 股（含数字或中文）"""
    return bool(re.search(r"\d", symbol)) or bool(re.search(r"[\u4e00-\u9fff]", symbol))


# ==================== Args Schemas ====================


class SearchStockNewsArgs(BaseModel):
    symbol: str = Field(description="股票代码或名称，如：AAPL、600519、贵州茅台")
    days: int = Field(default=7, description="搜索最近多少天的新闻，默认7天")
    count: int = Field(default=5, description="返回新闻条数，默认5条")


class AnalyzeNewsSentimentArgs(BaseModel):
    news_texts: List[str] = Field(description="新闻文本列表，每条为标题或摘要字符串")


class GetMarketSentimentArgs(BaseModel):
    market: str = Field(default="US", description="市场类型：US（美股）或 CN（A股）")
    count: int = Field(default=5, description="搜索新闻条数，默认5条")


class GetSectorSentimentArgs(BaseModel):
    sector: str = Field(
        description="行业/板块名称，如：人工智能、半导体、新能源、AI、Semiconductor"
    )
    count: int = Field(default=5, description="搜索新闻条数，默认5条")


# ==================== Tools ====================


@tool(args_schema=SearchStockNewsArgs)
async def search_stock_news(symbol: str, days: int = 7, count: int = 5) -> str:
    """
    搜索指定股票的近期新闻，返回格式化的新闻列表

    使用场景：
    - 获取某只股票的最新动态和媒体报道
    - 为后续情感分析提供原始文本数据

    Args:
        symbol: 股票代码或名称，如 AAPL、600519、贵州茅台
        days: 搜索最近多少天的新闻，默认7天
        count: 返回新闻条数，默认5条

    Returns:
        JSON 格式的新闻列表，包含标题、摘要、链接、来源等

    Example:
        search_stock_news("AAPL", days=7, count=5)
        search_stock_news("贵州茅台", days=30, count=10)
    """
    try:
        # 判断市场并构建查询
        if _is_chinese_stock(symbol):
            query = f"{symbol} 股票 新闻"
        else:
            query = f"{symbol} stock news"

        # 根据 days 设置 freshness
        if days <= 7:
            freshness = "oneWeek"
        elif days <= 30:
            freshness = "oneMonth"
        else:
            freshness = "noLimit"

        result = await web_search(
            query=query, freshness=freshness, summary=True, count=count
        )
        news_items = _extract_news_items(result)

        if not news_items:
            return json.dumps(
                {
                    "symbol": symbol,
                    "query": query,
                    "freshness": freshness,
                    "status": "empty",
                    "message": f"未找到 {symbol} 的近期新闻",
                    "news": [],
                },
                ensure_ascii=False,
                indent=2,
            )

        return json.dumps(
            {
                "symbol": symbol,
                "query": query,
                "freshness": freshness,
                "status": "success",
                "total": len(news_items),
                "news": news_items,
            },
            ensure_ascii=False,
            indent=2,
        )

    except Exception as e:
        return json.dumps(
            {
                "symbol": symbol,
                "status": "error",
                "message": f"搜索新闻失败: {str(e)}",
            },
            ensure_ascii=False,
        )


@tool(args_schema=AnalyzeNewsSentimentArgs)
def analyze_news_sentiment(news_texts: List[str]) -> str:
    """
    分析新闻文本的情感倾向（正面/负面/中性）

    当前使用关键词匹配作为 fallback，后续可无缝替换为专用情感模型。
    支持中英文混合文本。

    Args:
        news_texts: 新闻文本列表，每条为标题或摘要字符串

    Returns:
        JSON 格式的情感分析结果，包含总体情感、各文本得分等

    Example:
        analyze_news_sentiment(["AAPL财报超预期，股价大涨", "市场担忧衰退风险"])
    """
    try:
        if not news_texts:
            return json.dumps(
                {
                    "status": "error",
                    "message": "news_texts 不能为空列表",
                },
                ensure_ascii=False,
            )

        result = _keyword_sentiment_score(news_texts)
        result["status"] = "success"

        return json.dumps(result, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps(
            {
                "status": "error",
                "message": f"情感分析失败: {str(e)}",
            },
            ensure_ascii=False,
        )


@tool(args_schema=GetMarketSentimentArgs)
async def get_market_sentiment(market: str = "US", count: int = 5) -> str:
    """
    获取整体市场情绪，通过搜索市场-wide 新闻并分析情感倾向

    使用场景：
    - 判断当前大盘情绪是乐观还是悲观
    - 辅助择时决策

    Args:
        market: 市场类型，US（美股）或 CN（A股），默认 US
        count: 搜索新闻条数，默认5条

    Returns:
        JSON 格式的市场情绪摘要，包含新闻列表和情感分析

    Example:
        get_market_sentiment("US", count=5)
        get_market_sentiment("CN", count=10)
    """
    try:
        if market.upper() == "CN":
            query = "A股市场 走势 情绪"
            freshness = "oneWeek"
        else:
            query = "US stock market sentiment outlook"
            freshness = "oneWeek"

        search_result = await web_search(
            query=query, freshness=freshness, summary=True, count=count
        )
        news_items = _extract_news_items(search_result)

        if not news_items:
            return json.dumps(
                {
                    "market": market.upper(),
                    "status": "empty",
                    "message": f"未找到 {market.upper()} 市场的近期新闻",
                    "sentiment": None,
                    "news": [],
                },
                ensure_ascii=False,
                indent=2,
            )

        # 合并标题和摘要进行情感分析
        texts = [f"{item['title']} {item['summary']}" for item in news_items]
        sentiment = _keyword_sentiment_score(texts)

        return json.dumps(
            {
                "market": market.upper(),
                "status": "success",
                "query": query,
                "sentiment": {
                    "overall": sentiment["overall_sentiment"],
                    "score": sentiment["overall_score"],
                    "counts": sentiment["counts"],
                },
                "news": news_items,
            },
            ensure_ascii=False,
            indent=2,
        )

    except Exception as e:
        return json.dumps(
            {
                "market": market.upper(),
                "status": "error",
                "message": f"获取市场情绪失败: {str(e)}",
            },
            ensure_ascii=False,
        )


@tool(args_schema=GetSectorSentimentArgs)
async def get_sector_sentiment(sector: str, count: int = 5) -> str:
    """
    获取特定行业/板块的市场情绪，通过搜索行业新闻并分析情感倾向

    使用场景：
    - 判断某个行业（如 AI、半导体、新能源）当前的市场情绪
    - 辅助行业配置和板块轮动决策

    Args:
        sector: 行业/板块名称，如：人工智能、半导体、新能源、AI、Semiconductor
        count: 搜索新闻条数，默认5条

    Returns:
        JSON 格式的行业情绪摘要，包含新闻列表和情感分析

    Example:
        get_sector_sentiment("人工智能", count=5)
        get_sector_sentiment("Semiconductor", count=10)
    """
    try:
        # 判断中英文并构建查询
        if _is_chinese_stock(sector):
            query = f"{sector} 板块 新闻 走势"
        else:
            query = f"{sector} sector news outlook"

        freshness = "oneWeek"

        search_result = await web_search(
            query=query, freshness=freshness, summary=True, count=count
        )
        news_items = _extract_news_items(search_result)

        if not news_items:
            return json.dumps(
                {
                    "sector": sector,
                    "status": "empty",
                    "message": f"未找到 {sector} 板块的近期新闻",
                    "sentiment": None,
                    "news": [],
                },
                ensure_ascii=False,
                indent=2,
            )

        # 合并标题和摘要进行情感分析
        texts = [f"{item['title']} {item['summary']}" for item in news_items]
        sentiment = _keyword_sentiment_score(texts)

        return json.dumps(
            {
                "sector": sector,
                "status": "success",
                "query": query,
                "sentiment": {
                    "overall": sentiment["overall_sentiment"],
                    "score": sentiment["overall_score"],
                    "counts": sentiment["counts"],
                },
                "news": news_items,
            },
            ensure_ascii=False,
            indent=2,
        )

    except Exception as e:
        return json.dumps(
            {
                "sector": sector,
                "status": "error",
                "message": f"获取行业情绪失败: {str(e)}",
            },
            ensure_ascii=False,
        )


# ==================== 工具列表 ====================

SENTIMENT_TOOLS = [
    search_stock_news,
    analyze_news_sentiment,
    get_market_sentiment,
    get_sector_sentiment,
]


# ==================== 测试 ====================

if __name__ == "__main__":

    async def _test():
        # 测试 analyze_news_sentiment（同步工具）
        print("=== Test analyze_news_sentiment ===")
        texts = [
            "AAPL财报超预期，股价大涨",
            "市场担忧衰退风险，科技股下跌",
            "美联储维持利率不变",
        ]
        result = analyze_news_sentiment.invoke({"news_texts": texts})
        print(result)
        print()

        # 测试 search_stock_news
        print("=== Test search_stock_news ===")
        result = await search_stock_news.ainvoke(
            {"symbol": "AAPL", "days": 7, "count": 3}
        )
        print(result[:1000] if isinstance(result, str) else result)
        print()

        # 测试 get_market_sentiment
        print("=== Test get_market_sentiment ===")
        result = await get_market_sentiment.ainvoke({"market": "US", "count": 3})
        print(result[:1000] if isinstance(result, str) else result)
        print()

        # 测试 get_sector_sentiment
        print("=== Test get_sector_sentiment ===")
        result = await get_sector_sentiment.ainvoke({"sector": "AI", "count": 3})
        print(result[:1000] if isinstance(result, str) else result)

    asyncio.run(_test())
