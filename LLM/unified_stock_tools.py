"""
统一的Agent工具 - 基于Data.service
自动识别美股/A股，无需区分工具
"""

import asyncio
import json
import sys
import os
from datetime import datetime, timedelta

# 添加项目根目录到 Python 路径（确保能导入 Data、RAG 等模块）
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from langchain.tools import tool
from pydantic import BaseModel, Field
from typing import List, Optional

from Data.service import InfoService, fetch_stock_data
from Data.models import DataType

try:
    from Data.providers import us_stock, zh_stock, web_search
except ImportError:
    from Data.providers import us_stock, zh_stock, web_search

# 全局service实例（复用缓存）
_stock_service = InfoService(cache_ttl=300)


class StockPriceArgs(BaseModel):
    symbol: str = Field(description="股票代码或名称，如：AAPL、600519、贵州茅台")
    days: int = Field(default=30, description="获取多少天的数据，默认30天")


class StockInfoArgs(BaseModel):
    symbol: str = Field(description="股票代码或名称，如：AAPL、600519、贵州茅台")


class WebSearchArgs(BaseModel):
    query: str = Field(description="搜索内容")
    freshness: str = Field(
        default="day",
        description="搜可填值 noLimit：不限（默认）、oneDay：一天内、oneWeek一周内、oneMonth：一个月内、oneYear：一年内",
    )
    count: int = Field(default=5, description="返回多少条结果，默认5条")
    summary: bool = Field(default=True, description="是否返回摘要，默认True")


@tool(args_schema=StockPriceArgs)
async def get_stock_price(
    symbol: str, days: int = 30, as_of_date: Optional[str] = None
) -> str:
    """
    获取股票历史价格数据（自动识别美股/A股）

    支持输入格式：
    - 美股：AAPL, TSLA, BRK-B
    - A股代码：600519, 000001, sh600519
    - A股名称：贵州茅台, 平安银行

    返回包含日期、开盘、最高、最低、收盘、成交量的JSON数据
    """
    try:
        # 只请求必要区间的数据，避免一次性拉取过多历史数据
        end_date = (
            datetime.strptime(as_of_date, "%Y-%m-%d").date()
            if as_of_date
            else datetime.now().date()
        )
        start_date = end_date - timedelta(days=max(days * 2, 14))
        result = await _stock_service.fetch(
            symbol,
            [DataType.PRICE_DATA],
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
        )

        if not result.price_data:
            return f"错误：无法获取 {result.symbol} 的价格数据"

        recent = (
            result.price_data[-days:]
            if len(result.price_data) > days
            else result.price_data
        )
        prices = [
            {
                "date": p.date,
                "open": p.open,
                "high": p.high,
                "low": p.low,
                "close": p.close,
                "volume": p.volume,
            }
            for p in recent
        ]

        import json

        return json.dumps(
            {
                "symbol": result.symbol,
                "market_type": result.market_type,
                "currency": "USD" if result.market_type == "US" else "CNY",
                "data": prices,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )

    except Exception as e:
        return f"错误：{str(e)}"


@tool(args_schema=StockInfoArgs)
async def get_stock_basic_info(symbol: str) -> str:
    """
    获取股票基础信息（自动识别美股/A股）

    返回：公司名称、交易所、货币、市场类型
    """
    try:
        result = await _stock_service.fetch(symbol, [DataType.BASIC_INFO])

        if not result.basic_info:
            return f"错误：无法获取 {result.symbol} 的基础信息"

        import json

        return json.dumps(
            {
                "symbol": result.basic_info.symbol,
                "name": result.basic_info.name,
                "exchange": result.basic_info.exchange,
                "currency": result.basic_info.currency,
                "market_type": result.basic_info.market_type,
                "sector": result.basic_info.sector,
                "industry": result.basic_info.industry,
            },
            ensure_ascii=False,
            indent=2,
        )

    except Exception as e:
        return f"错误：{str(e)}"


@tool(args_schema=StockInfoArgs)
async def get_stock_company_info(symbol: str) -> str:
    """
    获取公司详细信息（业务描述、CEO、总部等）
    主要用于基本面分析
    """
    try:
        result = await _stock_service.fetch(
            symbol, [DataType.BASIC_INFO, DataType.COMPANY_DESCRIPTION]
        )

        import json

        data = {
            "symbol": result.symbol,
            "market_type": result.market_type,
        }

        if result.basic_info:
            data["name"] = result.basic_info.name
            data["exchange"] = result.basic_info.exchange

        if result.company_description:
            desc = result.company_description
            data["description"] = desc.description
            data["ceo"] = desc.ceo
            data["headquarters"] = desc.headquarters
            data["founded"] = desc.founded
        else:
            data["description"] = "暂无公司描述"

        return json.dumps(data, ensure_ascii=False, indent=2)

    except Exception as e:
        return f"错误：{str(e)}"


@tool(args_schema=StockInfoArgs)
async def get_stock_financial_report_links(symbol: str) -> str:
    """
    获取财务报告链接（用于下载完整报告）
    - 美股：返回10-K、10-Q的SEC官方链接
    - A股：返回财报查看链接

    适合需要查看完整PDF报告的场景
    """
    try:
        result = await _stock_service.fetch(symbol, [DataType.FINANCIAL_REPORT])

        import json

        reports = [
            {
                "type": r.report_type,
                "filing_date": r.filing_date,
                "url": r.url,
                "title": r.title,
            }
            for r in result.financial_reports
        ]

        return json.dumps(
            {
                "symbol": result.symbol,
                "market_type": result.market_type,
                "reports": reports,
            },
            ensure_ascii=False,
            indent=2,
        )

    except Exception as e:
        return f"错误：{str(e)}"


@tool(args_schema=StockInfoArgs)
async def get_stock_financial_statements(symbol: str) -> str:
    """
    获取财务报表数据（收入表、资产负债表、现金流量表）
    - 美股：从最新10-Q提取三张财务报表的具体数据
    - A股：返回财务摘要（营收、利润、资产负债等）

    返回具体财务数字，用于分析和计算财务指标
    """
    try:
        # 先获取基础信息确定市场类型
        result = await _stock_service.fetch(symbol, [DataType.BASIC_INFO])
        market_type = result.market_type
        symbol_std = result.symbol

        if market_type == "US":
            # 美股：提取10-Q中的财务表格
            categorized_tables, all_tables = await us_stock.get_financial_report(
                symbol_std
            )

            if not categorized_tables:
                return f"无法提取 {symbol_std} 的财务数据，可能是SEC网站访问限制或报告格式问题。"

            # 只保留关键的财务报表，过滤无关表格
            KEY_TABLES = ["Income_Statement", "Balance_Sheet", "Cash_Flow_Statement"]
            reports = {}

            for table_name, df in categorized_tables.items():
                # 只保留关键报表
                is_key_table = any(key in table_name for key in KEY_TABLES)
                if not is_key_table:
                    continue

                # 限制数据量：只取前5行，避免超出模型上下文
                reports[table_name] = {
                    "columns": (
                        df.columns.tolist()[:10] if hasattr(df, "columns") else []
                    ),  # 限制列数
                    "data": df.head(5).values.tolist() if hasattr(df, "values") else [],
                    "shape": (
                        [len(df), len(df.columns)] if hasattr(df, "columns") else [0, 0]
                    ),
                }

            # 如果没找到关键报表，返回提示
            if not reports:
                return json.dumps(
                    {
                        "symbol": symbol_std,
                        "market_type": "US",
                        "warning": "未识别出标准财务报表，原始数据包含多个表格",
                        "available_tables": list(categorized_tables.keys())[
                            :5
                        ],  # 只显示前5个表名
                    },
                    ensure_ascii=False,
                    indent=2,
                )

            return json.dumps(
                {
                    "symbol": symbol_std,
                    "market_type": "US",
                    "source": "SEC 10-Q Filing",
                    "reports": reports,
                    "note": f"已提取{len(reports)}张关键财务报表（收入表/资产负债表/现金流量表）",
                },
                ensure_ascii=False,
                indent=2,
            )

        else:
            # A股：获取财务摘要
            df = await zh_stock.get_financial_report(symbol_std)

            if df is None or df.empty:
                return f"无法获取 {symbol_std} 的财务数据"

            # 转换为JSON格式
            records = (
                df.head(20).to_dict(orient="records") if hasattr(df, "to_dict") else []
            )

            return json.dumps(
                {
                    "symbol": symbol_std,
                    "market_type": "CN",
                    "source": "新浪财经",
                    "data": records,
                    "note": "A股财务摘要数据",
                },
                ensure_ascii=False,
                indent=2,
            )

    except Exception as e:
        return f"错误：{str(e)}"


@tool(args_schema=StockInfoArgs)
async def get_full_stock_analysis(symbol: str) -> str:
    """
    获取股票完整信息（价格+基础信息+公司描述+财务报告）
    适合需要全面分析的场景
    """
    try:
        result = await _stock_service.fetch(
            symbol,
            [
                DataType.BASIC_INFO,
                DataType.PRICE_DATA,
                DataType.COMPANY_DESCRIPTION,
                DataType.FINANCIAL_REPORT,
            ],
        )

        # 使用已有的to_dict方法
        return result.to_json()

    except Exception as e:
        return f"错误：{str(e)}"


# ==================== RAG 检索工具 ====================

# 尝试导入 RAG 模块
try:
    from RAG.db_operations import VectorStoreBase

    _rag_available = True
except ImportError:
    VectorStoreBase = None
    _rag_available = False

# 全局 RAG 检索实例（懒加载）
_rag_store = None


def _get_rag_store():
    """获取或初始化 RAG 检索实例（单例模式）"""
    global _rag_store
    if _rag_store is None and _rag_available:
        try:
            _rag_store = VectorStoreBase()
            print("[RAG Tools] 向量数据库初始化成功")
        except Exception as e:
            print(f"[RAG Tools] 向量数据库初始化失败: {e}")
            _rag_store = None
    return _rag_store


class RAGSearchArgs(BaseModel):
    query: str = Field(description="搜索查询文本，如：什么是ROE、如何分析市盈率")
    top_k: int = Field(default=3, description="返回最相关的结果数量，默认3条，最大5条")


@tool(args_schema=RAGSearchArgs)
def search_financial_knowledge(query: str, top_k: int = 3) -> str:
    """
    检索金融知识库，获取与查询相关的专业金融知识

    使用场景：
    - 需要解释财务指标（ROE、PE、PB、现金流等）的含义和计算方法
    - 需要了解财务报表分析方法和技巧
    - 需要获取投资策略和估值方法的相关知识
    - 需要理解行业分析框架和投资理念

    Args:
        query: 搜索查询，描述你想了解的金融知识
        top_k: 返回结果数量（1-5），默认3条

    Returns:
        JSON格式的检索结果，包含相关问答对和相似度分数

    Example:
        query="什么是市盈率，如何判断高低" -> 返回PE定义、计算公式、判断标准等
        query="ROE是什么意思" -> 返回ROE定义、杜邦分析法、使用注意事项等
    """
    try:
        if not _rag_available:
            return json.dumps(
                {
                    "status": "error",
                    "message": "RAG 模块未安装或不可用",
                    "query": query,
                },
                ensure_ascii=False,
            )

        rag_store = _get_rag_store()

        if rag_store is None:
            return json.dumps(
                {
                    "status": "error",
                    "message": "RAG 知识库未初始化，请检查向量数据库是否存在",
                    "query": query,
                },
                ensure_ascii=False,
            )

        # 限制 top_k 范围
        top_k = max(1, min(5, top_k))

        # 执行检索
        results = rag_store.search(query, top_k=top_k)

        # 格式化结果
        formatted_results = []
        if results and results.get("documents") and len(results["documents"]) > 0:
            docs = results["documents"][0]
            metadatas = (
                results.get("metadatas", [[]])[0]
                if results.get("metadatas")
                else [{}] * len(docs)
            )
            distances = (
                results.get("distances", [[]])[0]
                if results.get("distances")
                else [0] * len(docs)
            )

            for i, (doc, meta, distance) in enumerate(zip(docs, metadatas, distances)):
                # 计算相似度（余弦相似度：1 - distance）
                similarity = 1 - distance if distance is not None else 0

                formatted_results.append(
                    {
                        "rank": i + 1,
                        "content": doc,
                        "similarity": round(similarity, 4),
                        "metadata": meta,
                    }
                )

        if not formatted_results:
            return json.dumps(
                {
                    "status": "empty",
                    "message": f"未找到与 '{query}' 相关的金融知识",
                    "query": query,
                    "suggestions": [
                        "尝试使用不同的关键词",
                        "使用更通用的金融术语",
                        "检查查询是否拼写正确",
                    ],
                },
                ensure_ascii=False,
            )

        return json.dumps(
            {
                "status": "success",
                "query": query,
                "total_results": len(formatted_results),
                "results": formatted_results,
            },
            ensure_ascii=False,
            indent=2,
        )

    except Exception as e:
        return json.dumps(
            {
                "status": "error",
                "message": f"检索过程发生错误: {str(e)}",
                "query": query,
            },
            ensure_ascii=False,
        )


class IndicatorExplanationArgs(BaseModel):
    indicator_name: str = Field(
        description="财务指标名称，如：ROE、PE、PB、毛利率、净利率、EPS等"
    )
    top_k: int = Field(default=2, description="返回结果数量，默认2条，最大5条")


@tool(args_schema=IndicatorExplanationArgs)
def get_financial_indicator_explanation(indicator_name: str, top_k: int = 2) -> str:
    """
    获取财务指标的详细解释，包括定义、计算方法和分析意义

    专门用于查询特定财务指标的含义，适合在分析过程中需要
    引用专业财务知识时使用。

    Args:
        indicator_name: 财务指标名称，如：ROE、PE、PB、毛利率、净利率、EPS等
        top_k: 返回结果数量，默认2条

    Returns:
        该指标的详细解释和应用场景

    Example:
        indicator_name="ROE" -> 净资产收益率的定义和杜邦分析法
        indicator_name="市盈率" -> PE的计算方法和估值判断标准
    """
    # 构建更精确的查询
    query = f"{indicator_name} 是什么意思 如何计算 怎么分析"
    return search_financial_knowledge.func(query, top_k)


@tool(args_schema=WebSearchArgs)
async def bocha_search(
    query: str, freshness: Optional[str] = None, summary: bool = True, count: int = 5
) -> str:
    """
    使用 Bocha API 进行网络搜索，获取最新的金融信息和新闻

    Args:
        query: 搜索查询文本，如：最新的AAPL财报、当前的市场趋势等
        freshness: 可选，指定结果的新鲜度，如：day、week、month
        summary: 是否返回结果摘要，默认为True
        count: 返回结果数量，默认5条，最大10条

    Returns:
        JSON格式的搜索结果，包括标题、链接、摘要等信息
    """
    try:
        result = await web_search.search(
            query=query, freshness=freshness, summary=summary, count=count
        )
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps(
            {"status": "error", "message": f"网络搜索失败: {str(e)}", "query": query},
            ensure_ascii=False,
        )


# ==================== 工具列表 ====================

UNIFIED_STOCK_TOOLS = [
    get_stock_price,
    get_stock_basic_info,
    get_stock_company_info,
    get_stock_financial_report_links,  # 获取财报链接（SEC/交易所）
    get_stock_financial_statements,  # 获取财务报表数据（三张表）
    get_full_stock_analysis,
    search_financial_knowledge,  # RAG检索金融知识
    get_financial_indicator_explanation,  # RAG获取财务指标解释
    bocha_search,  # 网络搜索工具
]


# ==================== 测试 ====================

if __name__ == "__main__":
    test = asyncio.run(
        bocha_search.ainvoke(
            {"query": "AAPL最新财报", "freshness": "day", "summary": True, "count": 3}
        )
    )
    print(test)
