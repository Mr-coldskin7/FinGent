"""
Data Models - 统一数据模型
定义所有数据类型的标准格式
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime


class DataType(Enum):
    """数据类型枚举"""
    BASIC_INFO = "basic_info"              # 基础信息（名称、交易所等）
    PRICE_DATA = "price_data"              # 价格数据（OHLCV）
    COMPANY_DESCRIPTION = "company_description"  # 公司描述
    FINANCIAL_REPORT = "financial_report"  # 财务报告
    NEWS_DATA = "news_data"                # 新闻数据
    TECHNICAL_INDICATORS = "technical_indicators"  # 技术指标


@dataclass
class BasicInfo:
    """股票基础信息"""
    symbol: str                    # 股票代码
    name: str                      # 公司名称
    currency: str                  # 货币（USD/CNY）
    exchange: str                  # 交易所
    exchange_full_name: str        # 交易所全称
    market_type: str               # 市场类型：US/CN/HK
    sector: Optional[str] = None   # 行业
    industry: Optional[str] = None # 细分行业
    country: Optional[str] = None  # 国家
    website: Optional[str] = None  # 官网
    market_cap: Optional[float] = None  # 市值


@dataclass
class PriceData:
    """价格数据（OHLCV）"""
    symbol: str
    date: str                      # 日期 YYYY-MM-DD
    open: float
    high: float
    low: float
    close: float
    volume: int
    adjusted_close: Optional[float] = None  # 调整后收盘价
    market_type: str = ""          # US/CN


@dataclass
class CompanyDescription:
    """公司描述信息"""
    symbol: str
    description: str               # 公司简介
    ceo: Optional[str] = None      # CEO
    employees: Optional[int] = None  # 员工数
    headquarters: Optional[str] = None  # 总部地址
    founded: Optional[str] = None  # 成立时间
    business_segments: Optional[List[Dict[str, Any]]] = field(default_factory=list)
    market_type: str = ""          # US/CN


@dataclass
class FinancialReport:
    """财务报告"""
    symbol: str
    report_type: str               # 报告类型：10-K, 10-Q, annual, quarterly
    filing_date: str               # filing日期
    url: str                       # 报告URL（美股）或标识（A股）
    title: Optional[str] = None    # 报告标题
    period_ended: Optional[str] = None  # 报告期结束日
    market_type: str = ""          # US/CN


@dataclass
class NewsData:
    """新闻数据"""
    symbol: str
    title: str                     # 标题
    summary: str                   # 摘要
    url: str                       # 链接
    published_date: str            # 发布日期
    source: str                    # 来源
    sentiment: Optional[str] = None  # 情感：positive/negative/neutral
    market_type: str = ""          # US/CN


@dataclass
class TechnicalIndicators:
    """技术指标"""
    symbol: str
    date: str
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    ema_12: Optional[float] = None
    ema_26: Optional[float] = None
    rsi: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    bollinger_upper: Optional[float] = None
    bollinger_lower: Optional[float] = None
    market_type: str = ""


@dataclass
class DataSource:
    """数据来源溯源信息"""
    source_type: str               # API/Cache/Calculated
    provider: str                  # 数据提供商：Tiingo/FMP/akshare/SEC
    timestamp: datetime            # 获取时间
    query_params: Dict[str, Any]   # 查询参数
    raw_response_hash: Optional[str] = None  # 原始响应哈希（用于校验）
    duration_ms: Optional[int] = None  # 耗时（毫秒）
    success: bool = True           # 是否成功
    error_msg: Optional[str] = None  # 错误信息


@dataclass
class UnifiedStockData:
    """
    统一股票数据容器
    包含所有可能的数据类型，根据需求填充
    """
    symbol: str
    market_type: str               # US/CN
    
    # 各类数据
    basic_info: Optional[BasicInfo] = None
    price_data: List[PriceData] = field(default_factory=list)
    company_description: Optional[CompanyDescription] = None
    financial_reports: List[FinancialReport] = field(default_factory=list)
    news_data: List[NewsData] = field(default_factory=list)
    technical_indicators: List[TechnicalIndicators] = field(default_factory=list)
    
    # 元数据
    created_at: datetime = field(default_factory=datetime.now)
    data_sources: List[DataSource] = field(default_factory=list)  # 溯源记录
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（用于序列化/返回给Agent）"""
        result = {
            "symbol": self.symbol,
            "market_type": self.market_type,
            "created_at": self.created_at.isoformat(),
            "data_summary": {
                "has_basic_info": self.basic_info is not None,
                "price_records": len(self.price_data),
                "has_description": self.company_description is not None,
                "financial_reports": len(self.financial_reports),
                "news_count": len(self.news_data),
            }
        }
        
        # 详细信息
        if self.basic_info:
            result["basic_info"] = {
                "symbol": self.basic_info.symbol,
                "name": self.basic_info.name,
                "exchange": self.basic_info.exchange,
                "currency": self.basic_info.currency,
            }
        
        if self.price_data:
            # 只返回最近几条，避免数据过大
            recent = self.price_data[-5:] if len(self.price_data) > 5 else self.price_data
            result["recent_prices"] = [
                {
                    "date": p.date,
                    "open": p.open,
                    "high": p.high,
                    "low": p.low,
                    "close": p.close,
                    "volume": p.volume
                }
                for p in recent
            ]
        
        if self.company_description:
            result["description"] = self.company_description.description[:500]  # 截断
        
        return result
    
    def to_json(self) -> str:
        """转换为 JSON 字符串"""
        import json
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
