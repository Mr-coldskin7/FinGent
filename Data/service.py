"""
Data Service - 统一数据服务入口
职责：市场识别 → 路由 → 获取 → 标准化 → 缓存
"""

import time
from typing import List, Optional, Tuple
from datetime import datetime

# 导入模型
from Data.models import DataType, UnifiedStockData, DataSource
from Data.cache import CacheManager

# 导入原始数据提供商
from Data.providers import us_stock, zh_stock
from Data.providers.adapters import (
    adapt_us_basic, adapt_us_prices, adapt_us_description, adapt_us_reports,
    adapt_cn_basic, adapt_cn_prices, adapt_cn_financial
)


class InfoService:
    """
    统一数据服务
    
    使用示例：
        service = InfoService()
        data = service.fetch("AAPL", [DataType.PRICE_DATA, DataType.BASIC_INFO])
        print(data.basic_info.name)
        print(data.price_data[-1].close)
    """
    
    def __init__(self, cache_ttl: int = 300):
        """
        初始化
        
        Args:
            cache_ttl: 缓存过期时间（秒），默认5分钟
        """
        self.cache = CacheManager(default_ttl=cache_ttl)
    
    def fetch(
        self, 
        identifier: str, 
        data_types: List[DataType],
        use_cache: bool = True,
        force_refresh: bool = False
    ) -> UnifiedStockData:
        """
        统一数据获取入口
        
        Args:
            identifier: 股票代码或名称（AAPL, 600519, 茅台）
            data_types: 需要的数据类型列表
            use_cache: 是否使用缓存
            force_refresh: 强制刷新缓存
        
        Returns:
            UnifiedStockData: 统一格式的股票数据
        """
        # 1. 解析市场类型和标准化代码
        market, symbol = self._resolve(identifier)
        
        # 2. 检查缓存
        if use_cache and not force_refresh:
            cached = self.cache.get(symbol, data_types)
            if cached:
                return cached
        
        # 3. 根据市场路由到对应获取器
        if market == 'US':
            data = self._fetch_us(symbol, data_types)
        elif market == 'CN':
            data = self._fetch_cn(symbol, data_types)
        else:
            raise ValueError(f"不支持的市场类型: {market}")
        
        # 4. 写入缓存
        if use_cache:
            self.cache.set(symbol, data_types, data)
        
        return data
    
    def _resolve(self, identifier: str) -> Tuple[str, str]:
        """
        解析市场类型和标准化代码
        
        Returns:
            Tuple[market, symbol]: 市场类型(US/CN)和标准化代码
        """
        identifier = identifier.strip()
        
        # A股：中文名称（最先判断，避免纯中文被误判）
        # 如果包含中文字符，直接尝试A股转换
        import re
        if re.search(r'[\u4e00-\u9fff]', identifier):
            try:
                code = zh_stock.input2number(identifier)
                return 'CN', code
            except Exception as e:
                raise ValueError(f"无法识别A股名称: {identifier}, 错误: {e}")
        
        # 美股：纯字母，1-5位（允许包含-，如BRK-B）
        clean_id = identifier.replace('-', '').replace('.', '')
        if clean_id.isalpha() and 1 <= len(clean_id) <= 5:
            return 'US', identifier.upper()
        
        # A股：带前缀（sh600519, sz000001）
        id_upper = identifier.upper()
        if id_upper.startswith(('SH', 'SZ', 'BJ')) and len(identifier) == 8:
            return 'CN', identifier[2:]
        
        # A股：6位数字
        if identifier.isdigit():
            if len(identifier) == 6:
                return 'CN', identifier
            # 可能是港股或其他，暂不支持
            raise ValueError(f"暂不支持的股票代码格式: {identifier}")
        
        raise ValueError(f"无法识别股票代码: {identifier}")
    
    def _fetch_us(self, symbol: str, data_types: List[DataType]) -> UnifiedStockData:
        """获取美股数据"""
        result = UnifiedStockData(symbol=symbol, market_type='US')
        sources = []
        
        for dtype in data_types:
            source = DataSource(
                source_type='API',
                provider='',
                query_params={'symbol': symbol, 'type': dtype.value},
                timestamp=datetime.now(),
                success=False
            )
            
            try:
                start_time = time.time()
                
                if dtype == DataType.BASIC_INFO:
                    raw = us_stock.get_basic_info_by_symbol(symbol)
                    result.basic_info = adapt_us_basic(raw, symbol)
                    source.provider = 'FMP'
                    source.success = True
                    
                elif dtype == DataType.PRICE_DATA:
                    raw = us_stock.get_historical_stock_price_by_symbol(symbol)
                    result.price_data = adapt_us_prices(raw, symbol)
                    source.provider = 'Tiingo'
                    source.success = True
                    
                elif dtype == DataType.COMPANY_DESCRIPTION:
                    raw = us_stock.get_description(symbol)
                    result.company_description = adapt_us_description(raw, symbol)
                    source.provider = 'Tiingo'
                    source.success = True
                    
                elif dtype == DataType.FINANCIAL_REPORT:
                    k_urls = us_stock.get_10K_financial_report(symbol, 1)
                    q_urls = us_stock.get_10Q_financial_report(symbol, 1)
                    result.financial_reports = adapt_us_reports(k_urls, q_urls, symbol)
                    source.provider = 'SEC'
                    source.success = True
                
                source.duration_ms = int((time.time() - start_time) * 1000)
                
            except Exception as e:
                source.error_msg = str(e)
                # 继续获取其他数据类型，不中断
            
            sources.append(source)
        
        result.data_sources = sources
        return result
    
    def _fetch_cn(self, symbol: str, data_types: List[DataType]) -> UnifiedStockData:
        """获取A股数据"""
        result = UnifiedStockData(symbol=symbol, market_type='CN')
        sources = []
        
        for dtype in data_types:
            source = DataSource(
                source_type='API',
                provider='akshare',
                query_params={'symbol': symbol, 'type': dtype.value},
                timestamp=datetime.now(),
                success=False
            )
            
            try:
                start_time = time.time()
                
                if dtype == DataType.BASIC_INFO:
                    raw = zh_stock.stock_individual_info(symbol)
                    result.basic_info = adapt_cn_basic(raw, symbol)
                    source.success = True
                    
                elif dtype == DataType.PRICE_DATA:
                    df = zh_stock.get_historical_stock_price_by_symbol(symbol)
                    result.price_data = adapt_cn_prices(df, symbol)
                    source.success = True
                    
                elif dtype == DataType.COMPANY_DESCRIPTION:
                    # A股公司描述可以通过基本信息获取
                    raw = zh_stock.stock_individual_info(symbol)
                    if raw is not None and not raw.empty:
                        desc = f"公司简称: {raw.iloc[0].get('公司简称', '')}, 所属行业: {raw.iloc[0].get('所属行业', '')}"
                        from Data.models import CompanyDescription
                        result.company_description = CompanyDescription(
                            symbol=symbol,
                            description=desc,
                            market_type='CN'
                        )
                    source.success = True
                    
                elif dtype == DataType.FINANCIAL_REPORT:
                    raw = zh_stock.get_financial_report(symbol)
                    result.financial_reports = adapt_cn_financial(raw, symbol)
                    source.success = True
                
                source.duration_ms = int((time.time() - start_time) * 1000)
                
            except Exception as e:
                source.error_msg = str(e)
            
            sources.append(source)
        
        result.data_sources = sources
        return result
    
    def clear_cache(self, symbol: str = None):
        """清除缓存"""
        self.cache.invalidate(symbol)
    
    def get_cache_stats(self) -> dict:
        """获取缓存统计"""
        return self.cache.get_stats()


# 便捷函数：快速获取

def fetch_stock_data(identifier: str, data_types: List[DataType] = None) -> UnifiedStockData:
    """
    快速获取股票数据（使用默认缓存配置）
    
    Args:
        identifier: 股票代码或名称
        data_types: 数据类型列表，默认为 [BASIC_INFO, PRICE_DATA]
    
    Returns:
        UnifiedStockData
    """
    if data_types is None:
        data_types = [DataType.BASIC_INFO, DataType.PRICE_DATA]
    
    service = InfoService()
    return service.fetch(identifier, data_types)
