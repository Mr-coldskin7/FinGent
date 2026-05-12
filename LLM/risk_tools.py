"""
Risk Management and Position Sizing Tools
基于 Data.service 的风险管理工具集
提供 VaR、仓位管理、组合风险指标、相关性分析和流动性风险评估
"""

import asyncio
import json
import sys
import os
from datetime import datetime
from typing import List, Optional

import numpy as np
import pandas as pd

from langchain.tools import tool
from pydantic import BaseModel, Field

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from Data.service import InfoService, fetch_stock_data
from Data.models import DataType

# 全局 service 实例（复用缓存）
_risk_service = InfoService(cache_ttl=300)


# =============================================================================
# Pydantic Args Schemas
# =============================================================================


class CalculateVaRArgs(BaseModel):
    symbol: str = Field(description="股票代码或名称，如：AAPL、600519、贵州茅台")
    position_value: float = Field(description="持仓金额（对应货币单位）")
    confidence: float = Field(default=0.95, description="置信水平，默认0.95")
    lookback_days: int = Field(
        default=252, description="回看天数，默认252个交易日（约1年）"
    )


class CalculatePositionSizeArgs(BaseModel):
    symbol: str = Field(description="股票代码或名称，如：AAPL、600519、贵州茅台")
    portfolio_value: float = Field(description="组合总价值（对应货币单位）")
    risk_per_trade_pct: float = Field(
        default=0.02, description="每笔交易风险比例，默认0.02（2%）"
    )
    method: str = Field(
        default="kelly",
        description="仓位计算方法：kelly（凯利公式）或 fixed_fractional（固定比例）",
    )


class CalculatePortfolioRiskMetricsArgs(BaseModel):
    symbol: str = Field(description="股票代码或名称，如：AAPL、600519、贵州茅台")
    lookback_days: int = Field(default=252, description="回看天数，默认252个交易日")


class CalculateCorrelationMatrixArgs(BaseModel):
    symbols: List[str] = Field(
        description='股票代码列表，如：["AAPL", "MSFT", "GOOGL"] 或 ["600519", "000001"]'
    )
    lookback_days: int = Field(default=90, description="回看天数，默认90个交易日")


class AssessLiquidityRiskArgs(BaseModel):
    symbol: str = Field(description="股票代码或名称，如：AAPL、600519、贵州茅台")
    position_value: float = Field(description="持仓金额（对应货币单位）")
    lookback_days: int = Field(default=30, description="回看天数，默认30个交易日")


# =============================================================================
# Helper Functions
# =============================================================================


def _resolve_market(symbol: str) -> str:
    """解析市场类型（US / CN）"""
    identifier = symbol.strip()
    import re

    if re.search(r"[一-鿿]", identifier):
        return "CN"
    clean_id = identifier.replace("-", "").replace(".", "")
    if clean_id.isalpha() and 1 <= len(clean_id) <= 5:
        return "US"
    id_upper = identifier.upper()
    if id_upper.startswith(("SH", "SZ", "BJ")) and len(identifier) == 8:
        return "CN"
    if identifier.isdigit():
        if len(identifier) == 6:
            return "CN"
    return "US"


def _price_data_to_df(price_data: list) -> pd.DataFrame:
    """将 PriceData 列表转换为 pandas DataFrame"""
    if not price_data:
        return pd.DataFrame()
    df = pd.DataFrame(
        [
            {
                "date": p.date,
                "open": p.open,
                "high": p.high,
                "low": p.low,
                "close": p.close,
                "volume": p.volume,
            }
            for p in price_data
        ]
    )
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def _calculate_returns(df: pd.DataFrame) -> pd.Series:
    """计算日收益率（对数收益率）"""
    return np.log(df["close"] / df["close"].shift(1)).dropna()


def _calculate_max_drawdown(df: pd.DataFrame) -> float:
    """计算最大回撤"""
    cummax = df["close"].cummax()
    drawdown = (df["close"] - cummax) / cummax
    return drawdown.min()


def _annualized_volatility(returns: pd.Series, trading_days: int = 252) -> float:
    """年化波动率"""
    return returns.std() * np.sqrt(trading_days)


def _sharpe_ratio(
    returns: pd.Series, risk_free_rate: float = 0.03, trading_days: int = 252
) -> float:
    """夏普比率"""
    excess_return = returns.mean() * trading_days - risk_free_rate
    vol = returns.std() * np.sqrt(trading_days)
    if vol == 0:
        return 0.0
    return excess_return / vol


def _sortino_ratio(
    returns: pd.Series, risk_free_rate: float = 0.03, trading_days: int = 252
) -> float:
    """索提诺比率（只考虑下行波动）"""
    excess_return = returns.mean() * trading_days - risk_free_rate
    downside_returns = returns[returns < 0]
    downside_std = (
        downside_returns.std() * np.sqrt(trading_days)
        if len(downside_returns) > 0
        else 0
    )
    if downside_std == 0:
        return 0.0
    return excess_return / downside_std


def _calculate_beta(stock_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """计算 Beta 系数"""
    aligned = pd.concat([stock_returns, benchmark_returns], axis=1).dropna()
    if aligned.empty or len(aligned) < 2:
        return 1.0
    cov = aligned.cov().iloc[0, 1]
    bench_var = aligned.iloc[:, 1].var()
    if bench_var == 0:
        return 1.0
    return cov / bench_var


async def _fetch_price_data(symbol: str, lookback_days: int = 252) -> pd.DataFrame:
    """获取价格数据并转为 DataFrame"""
    result = await _risk_service.fetch(symbol, [DataType.PRICE_DATA])
    if not result.price_data:
        raise ValueError(f"无法获取 {result.symbol} 的价格数据")
    df = _price_data_to_df(result.price_data)
    if len(df) > lookback_days:
        df = df.tail(lookback_days).reset_index(drop=True)
    if len(df) < 30:
        raise ValueError(f"价格数据不足：仅获取到 {len(df)} 条记录，需要至少 30 条")
    return df


async def _fetch_benchmark_returns(
    market_type: str, lookback_days: int = 252
) -> Optional[pd.Series]:
    """获取基准指数收益率"""
    try:
        if market_type == "US":
            benchmark_symbol = "SPY"
        else:
            # A股使用沪深300，通过 akshare 或直接获取
            benchmark_symbol = "000300"  # 沪深300
        df = await _fetch_price_data(benchmark_symbol, lookback_days)
        return _calculate_returns(df)
    except Exception:
        return None


# =============================================================================
# Tool 1: calculate_var
# =============================================================================


@tool(args_schema=CalculateVaRArgs)
async def calculate_var(
    symbol: str,
    position_value: float,
    confidence: float = 0.95,
    lookback_days: int = 252,
) -> str:
    """
    计算股票持仓的历史模拟法 VaR（Value at Risk）

    通过获取历史价格数据，计算日收益率，基于历史分位数估计在给定置信水平下的最大可能损失。

    Args:
        symbol: 股票代码或名称，如 AAPL、600519、贵州茅台
        position_value: 持仓金额
        confidence: 置信水平，默认 0.95（即 95% VaR）
        lookback_days: 回看天数，默认 252 个交易日（约 1 年）

    Returns:
        JSON 字符串，包含 var_amount（VaR 金额）、var_pct（VaR 百分比）、confidence_level、lookback_days、method

    Example:
        calculate_var("AAPL", 100000, confidence=0.95, lookback_days=252)
        -> {"var_amount": 3250.5, "var_pct": 0.0325, "confidence_level": 0.95, ...}
    """
    try:
        if confidence <= 0 or confidence >= 1:
            return json.dumps(
                {"error": "置信水平必须在 (0, 1) 之间"}, ensure_ascii=False
            )
        if position_value <= 0:
            return json.dumps({"error": "持仓金额必须大于 0"}, ensure_ascii=False)

        df = await _fetch_price_data(symbol, lookback_days)
        returns = _calculate_returns(df)

        # 历史模拟法：取收益率的 (1 - confidence) 分位数
        var_pct = np.percentile(returns, (1 - confidence) * 100)
        var_amount = abs(var_pct) * position_value

        return json.dumps(
            {
                "symbol": symbol,
                "var_amount": round(var_amount, 2),
                "var_pct": round(abs(var_pct), 6),
                "confidence_level": confidence,
                "lookback_days": len(df),
                "method": "historical_simulation",
                "data_points": len(returns),
            },
            ensure_ascii=False,
            indent=2,
        )

    except Exception as e:
        return json.dumps({"error": f"计算 VaR 失败: {str(e)}"}, ensure_ascii=False)


# =============================================================================
# Tool 2: calculate_position_size
# =============================================================================


@tool(args_schema=CalculatePositionSizeArgs)
async def calculate_position_size(
    symbol: str,
    portfolio_value: float,
    risk_per_trade_pct: float = 0.02,
    method: str = "kelly",
) -> str:
    """
    基于凯利公式（Kelly Criterion）和固定比例法计算最优仓位规模

    获取历史价格数据，计算历史胜率（win rate）和平均盈亏比，
    然后应用凯利公式 f* = (p*b - q) / b 计算最优仓位比例。
    同时返回固定比例法的结果作为对比。

    Args:
        symbol: 股票代码或名称
        portfolio_value: 组合总价值
        risk_per_trade_pct: 每笔交易可承受风险比例，默认 0.02（2%）
        method: 计算方法，kelly（凯利公式）或 fixed_fractional（固定比例）

    Returns:
        JSON 字符串，包含 recommended_shares、position_value、position_pct、method_used、
        kelly_fraction、fixed_fractional_result

    Example:
        calculate_position_size("AAPL", 100000, risk_per_trade_pct=0.02, method="kelly")
    """
    try:
        if portfolio_value <= 0:
            return json.dumps({"error": "组合价值必须大于 0"}, ensure_ascii=False)
        if risk_per_trade_pct <= 0 or risk_per_trade_pct > 1:
            return json.dumps(
                {"error": "风险比例必须在 (0, 1] 之间"}, ensure_ascii=False
            )
        if method not in ("kelly", "fixed_fractional"):
            return json.dumps(
                {"error": "method 必须是 kelly 或 fixed_fractional"}, ensure_ascii=False
            )

        df = await _fetch_price_data(symbol, lookback_days=252)
        returns = _calculate_returns(df)

        # 计算历史胜率和平均盈亏
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]

        total_trades = len(positive_returns) + len(negative_returns)
        if total_trades == 0:
            return json.dumps({"error": "无法计算盈亏数据"}, ensure_ascii=False)

        win_rate = len(positive_returns) / total_trades
        loss_rate = 1 - win_rate

        avg_win = positive_returns.mean() if len(positive_returns) > 0 else 0
        avg_loss = abs(negative_returns.mean()) if len(negative_returns) > 0 else 0

        # 凯利公式：f* = (p*b - q) / b，其中 b = avg_win / avg_loss
        kelly_fraction = 0.0
        if avg_loss > 0 and avg_win > 0:
            b = avg_win / avg_loss
            if b > 0:
                kelly_fraction = (win_rate * b - loss_rate) / b
                kelly_fraction = max(0.0, min(kelly_fraction, 1.0))  # 限制在 [0, 1]

        # 固定比例法：risk_per_trade / avg_loss
        fixed_fractional_result = 0.0
        if avg_loss > 0:
            fixed_fractional_result = risk_per_trade_pct / avg_loss
            fixed_fractional_result = max(0.0, min(fixed_fractional_result, 1.0))

        # 根据 method 选择最终仓位比例
        if method == "kelly":
            position_pct = kelly_fraction
        else:
            position_pct = fixed_fractional_result

        # 保守建议：使用半凯利（half-kelly）作为推荐值，避免过度杠杆
        recommended_pct = position_pct * 0.5 if method == "kelly" else position_pct
        recommended_pct = max(0.0, min(recommended_pct, 1.0))

        position_value = portfolio_value * recommended_pct
        current_price = df["close"].iloc[-1]
        recommended_shares = (
            int(position_value / current_price) if current_price > 0 else 0
        )

        return json.dumps(
            {
                "symbol": symbol,
                "portfolio_value": round(portfolio_value, 2),
                "current_price": round(current_price, 4),
                "recommended_shares": recommended_shares,
                "position_value": round(recommended_shares * current_price, 2),
                "position_pct": round(recommended_pct, 6),
                "method_used": method,
                "kelly_fraction": round(kelly_fraction, 6),
                "fixed_fractional_result": round(fixed_fractional_result, 6),
                "win_rate": round(win_rate, 4),
                "avg_win": round(avg_win, 6),
                "avg_loss": round(avg_loss, 6),
                "risk_per_trade_pct": risk_per_trade_pct,
                "note": "推荐仓位已采用半凯利（half-kelly）保守估计",
            },
            ensure_ascii=False,
            indent=2,
        )

    except Exception as e:
        return json.dumps({"error": f"计算仓位规模失败: {str(e)}"}, ensure_ascii=False)


# =============================================================================
# Tool 3: calculate_portfolio_risk_metrics
# =============================================================================


@tool(args_schema=CalculatePortfolioRiskMetricsArgs)
async def calculate_portfolio_risk_metrics(
    symbol: str, lookback_days: int = 252
) -> str:
    """
    计算股票的综合风险指标

    包括：年化波动率、最大回撤、夏普比率（假设无风险利率 3%）、
    Beta 系数（相对 SPY 或沪深300）、索提诺比率（Sortino Ratio）。

    Args:
        symbol: 股票代码或名称
        lookback_days: 回看天数，默认 252 个交易日

    Returns:
        JSON 字符串，包含所有风险指标

    Example:
        calculate_portfolio_risk_metrics("AAPL", lookback_days=252)
    """
    try:
        df = await _fetch_price_data(symbol, lookback_days)
        returns = _calculate_returns(df)

        if len(returns) < 30:
            return json.dumps(
                {"error": "收益率数据不足，无法计算风险指标"}, ensure_ascii=False
            )

        market_type = _resolve_market(symbol)

        # 基础指标
        ann_vol = _annualized_volatility(returns)
        max_dd = _calculate_max_drawdown(df)
        sharpe = _sharpe_ratio(returns, risk_free_rate=0.03)
        sortino = _sortino_ratio(returns, risk_free_rate=0.03)

        # 计算总收益率和年化收益率
        total_return = (df["close"].iloc[-1] / df["close"].iloc[0]) - 1
        years = len(df) / 252
        ann_return = (
            (1 + total_return) ** (1 / max(years, 0.01)) - 1 if years > 0 else 0.0
        )

        # Beta 计算
        benchmark_returns = await _fetch_benchmark_returns(
            market_type, lookback_days=len(df)
        )
        beta = 1.0
        if benchmark_returns is not None and len(benchmark_returns) > 0:
            beta = _calculate_beta(returns, benchmark_returns)

        benchmark_name = "SPY" if market_type == "US" else "沪深300"

        return json.dumps(
            {
                "symbol": symbol,
                "market_type": market_type,
                "lookback_days": len(df),
                "annualized_volatility": round(ann_vol, 4),
                "annualized_return": round(ann_return, 4),
                "max_drawdown": round(max_dd, 4),
                "sharpe_ratio": round(sharpe, 4),
                "sortino_ratio": round(sortino, 4),
                "beta": round(beta, 4),
                "benchmark": benchmark_name,
                "risk_free_rate": 0.03,
                "data_points": len(returns),
            },
            ensure_ascii=False,
            indent=2,
        )

    except Exception as e:
        return json.dumps({"error": f"计算风险指标失败: {str(e)}"}, ensure_ascii=False)


# =============================================================================
# Tool 4: calculate_correlation_matrix
# =============================================================================


@tool(args_schema=CalculateCorrelationMatrixArgs)
async def calculate_correlation_matrix(
    symbols: List[str], lookback_days: int = 90
) -> str:
    """
    计算多只股票之间的收益率相关性矩阵

    获取每只股票的历史价格数据，计算日收益率，然后计算皮尔逊相关系数矩阵。
    同时计算组合分散化评分（diversification score）。

    Args:
        symbols: 股票代码列表，如 ["AAPL", "MSFT", "GOOGL"] 或 ["600519", "000001"]
        lookback_days: 回看天数，默认 90 个交易日

    Returns:
        JSON 字符串，包含相关性矩阵和分散化评分

    Example:
        calculate_correlation_matrix(["AAPL", "MSFT", "GOOGL"], lookback_days=90)
    """
    try:
        if not symbols or len(symbols) < 2:
            return json.dumps(
                {"error": "至少需要提供 2 只股票代码"}, ensure_ascii=False
            )

        returns_dict = {}
        failed_symbols = []

        for sym in symbols:
            try:
                df = await _fetch_price_data(sym, lookback_days)
                ret = _calculate_returns(df)
                if len(ret) >= 20:
                    returns_dict[sym] = ret
                else:
                    failed_symbols.append(sym)
            except Exception:
                failed_symbols.append(sym)

        if len(returns_dict) < 2:
            return json.dumps(
                {"error": f"成功获取数据的股票不足 2 只，失败: {failed_symbols}"},
                ensure_ascii=False,
            )

        # 构建收益率 DataFrame（对齐日期）
        returns_df = pd.DataFrame(returns_dict)
        returns_df = returns_df.dropna()

        if returns_df.empty or len(returns_df) < 10:
            return json.dumps({"error": "对齐后的收益率数据不足"}, ensure_ascii=False)

        corr_matrix = returns_df.corr()

        # 转换为字典格式
        corr_dict = {}
        for col in corr_matrix.columns:
            corr_dict[col] = {
                row: round(float(corr_matrix.loc[row, col]), 4)
                for row in corr_matrix.index
            }

        # 计算分散化评分：平均相关系数的补数（1 - avg_corr）
        # 只取上三角（不含对角线）
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        avg_corr = upper_tri.stack().mean()
        diversification_score = round(1 - avg_corr, 4)

        # 流动性评级
        if diversification_score >= 0.7:
            diversification_rating = "优秀"
        elif diversification_score >= 0.4:
            diversification_rating = "良好"
        else:
            diversification_rating = "较差"

        return json.dumps(
            {
                "symbols": list(returns_dict.keys()),
                "failed_symbols": failed_symbols if failed_symbols else None,
                "lookback_days": lookback_days,
                "aligned_data_points": len(returns_df),
                "correlation_matrix": corr_dict,
                "average_correlation": round(float(avg_corr), 4),
                "diversification_score": diversification_score,
                "diversification_rating": diversification_rating,
            },
            ensure_ascii=False,
            indent=2,
        )

    except Exception as e:
        return json.dumps(
            {"error": f"计算相关性矩阵失败: {str(e)}"}, ensure_ascii=False
        )


# =============================================================================
# Tool 5: assess_liquidity_risk
# =============================================================================


@tool(args_schema=AssessLiquidityRiskArgs)
async def assess_liquidity_risk(
    symbol: str, position_value: float, lookback_days: int = 30
) -> str:
    """
    评估股票持仓的流动性风险

    基于历史成交量和收盘价估算平均成交量、日均成交金额、
    预计平仓天数，并给出流动性评级（高/中/低）。

    Args:
        symbol: 股票代码或名称
        position_value: 持仓金额
        lookback_days: 回看天数，默认 30 个交易日

    Returns:
        JSON 字符串，包含 avg_volume、avg_dollar_volume、days_to_liquidate_estimate、liquidity_rating

    Example:
        assess_liquidity_risk("AAPL", 500000, lookback_days=30)
    """
    try:
        if position_value <= 0:
            return json.dumps({"error": "持仓金额必须大于 0"}, ensure_ascii=False)

        df = await _fetch_price_data(symbol, lookback_days)

        avg_volume = float(df["volume"].mean())
        avg_close = float(df["close"].mean())
        avg_dollar_volume = avg_volume * avg_close

        # 估算平仓天数（假设每天最多交易日均成交金额的 10% 而不产生显著冲击）
        daily_liquidation_capacity = avg_dollar_volume * 0.10
        if daily_liquidation_capacity > 0:
            days_to_liquidate = position_value / daily_liquidation_capacity
        else:
            days_to_liquidate = float("inf")

        # 流动性评级
        if days_to_liquidate <= 1:
            liquidity_rating = "高"
        elif days_to_liquidate <= 5:
            liquidity_rating = "中"
        else:
            liquidity_rating = "低"

        # 价格冲击估算（基于 Amihud 非流动性指标的简化版）
        # 使用 |收益率| / 成交金额 的日均值作为代理
        returns = _calculate_returns(df)
        dollar_volume_series = df["volume"] * df["close"]
        # 对齐长度
        aligned_vol = dollar_volume_series.iloc[-len(returns) :].reset_index(drop=True)
        amihud_proxy = (
            (returns.abs() / (aligned_vol / 1e6)).mean() if aligned_vol.sum() > 0 else 0
        )

        return json.dumps(
            {
                "symbol": symbol,
                "position_value": round(position_value, 2),
                "lookback_days": len(df),
                "avg_volume": int(avg_volume),
                "avg_close": round(avg_close, 4),
                "avg_dollar_volume": round(avg_dollar_volume, 2),
                "days_to_liquidate_estimate": round(days_to_liquidate, 2),
                "liquidity_rating": liquidity_rating,
                "price_impact_proxy": round(float(amihud_proxy), 8),
                "note": "价格冲击代理值越高表示流动性越差；平仓天数假设每日最多交易日均成交金额的10%",
            },
            ensure_ascii=False,
            indent=2,
        )

    except Exception as e:
        return json.dumps(
            {"error": f"评估流动性风险失败: {str(e)}"}, ensure_ascii=False
        )


# =============================================================================
# Tool List
# =============================================================================

RISK_TOOLS = [
    calculate_var,
    calculate_position_size,
    calculate_portfolio_risk_metrics,
    calculate_correlation_matrix,
    assess_liquidity_risk,
]


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":

    async def _test():
        print("=== Test calculate_var ===")
        r1 = await calculate_var.ainvoke({"symbol": "AAPL", "position_value": 100000})
        print(r1[:500])

        print("\n=== Test calculate_position_size ===")
        r2 = await calculate_position_size.ainvoke(
            {"symbol": "AAPL", "portfolio_value": 100000}
        )
        print(r2[:500])

        print("\n=== Test calculate_portfolio_risk_metrics ===")
        r3 = await calculate_portfolio_risk_metrics.ainvoke({"symbol": "AAPL"})
        print(r3[:500])

        print("\n=== Test calculate_correlation_matrix ===")
        r4 = await calculate_correlation_matrix.ainvoke({"symbols": ["AAPL", "MSFT"]})
        print(r4[:500])

        print("\n=== Test assess_liquidity_risk ===")
        r5 = await assess_liquidity_risk.ainvoke(
            {"symbol": "AAPL", "position_value": 500000}
        )
        print(r5[:500])

    asyncio.run(_test())
