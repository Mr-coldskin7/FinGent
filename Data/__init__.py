"""
Data - 统一数据访问层

职责：
1. 抽象不同数据源的差异（美股/A股）
2. 统一数据格式（转换为标准 dataclass）
3. 数据缓存
4. 数据来源溯源（Provenance）

使用示例：
    from Data import InfoService, DataType, fetch_stock_data
    
    # 方式1：使用服务类
    service = InfoService()
    data = service.fetch("AAPL", [DataType.BASIC_INFO, DataType.PRICE_DATA])
    
    # 方式2：便捷函数
    data = fetch_stock_data("600519", [DataType.PRICE_DATA])
    
    # 访问数据
    print(data.basic_info.name)
    print(data.price_data[-1].close)
    print(data.to_json())  # 转为JSON给Agent
"""

# 数据模型
from Data.models import (
    DataType,
    BasicInfo,
    PriceData,
    CompanyDescription,
    FinancialReport,
    NewsData,
    TechnicalIndicators,
    UnifiedStockData,
    DataSource
)

# 主服务
from Data.service import InfoService, fetch_stock_data

# 缓存（如需单独使用）
from Data.cache import CacheManager

__all__ = [
    # 模型
    'DataType',
    'BasicInfo',
    'PriceData',
    'CompanyDescription',
    'FinancialReport',
    'NewsData',
    'TechnicalIndicators',
    'UnifiedStockData',
    'DataSource',
    # 服务
    'InfoService',
    'fetch_stock_data',
    'CacheManager',
]

__version__ = '0.1.0'
