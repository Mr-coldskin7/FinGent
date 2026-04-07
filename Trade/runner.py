from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Any, Dict, Optional

import pandas as pd

from Data.models import DataType
from Data.service import InfoService
from Trade.adapter import build_graph_input, extract_graph_signal
from typing import List

try:
    import backtrader as bt
except ImportError as exc:
    bt = None
    _BACKTRADER_IMPORT_ERROR = exc
else:
    _BACKTRADER_IMPORT_ERROR = None


def _require_backtrader() -> None:
    if bt is None:
        raise ImportError(
            "backtrader is required. Install it with: pip install backtrader"
        ) from _BACKTRADER_IMPORT_ERROR


def load_price_dataframe(symbol: str, start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
    """加载价格数据，支持指定日期范围。
    
    为了避免缓存数据不完整的问题，当指定了日期范围时，
    直接调用底层 API 获取数据而不是使用 InfoService 的缓存。
    """
    from Data.providers import us_stock, zh_stock
    from Data.providers.adapters import adapt_us_prices, adapt_cn_prices
    from Data.service import InfoService  # 用于市场识别
    
    # 使用 InfoService 解析市场类型
    service = InfoService()
    market, normalized_symbol = service._resolve(symbol)
    
    # 根据市场类型调用对应的底层 API，传入日期范围
    if market == 'US':
        raw = us_stock.get_historical_stock_price_by_symbol(
            symbol=normalized_symbol,
            startDate=start,
            endDate=end
        )
        price_data = adapt_us_prices(raw, normalized_symbol)
    elif market == 'CN':
        # A股数据获取（假设 zh_stock 也有类似的日期参数支持）
        df = zh_stock.get_historical_stock_price_by_symbol(normalized_symbol)
        price_data = adapt_cn_prices(df, normalized_symbol)
    else:
        raise ValueError(f"不支持的市场类型: {market}")
    
    # 转换为 DataFrame
    rows = []
    for p in price_data:
        rows.append(
            {
                "datetime": pd.to_datetime(str(p.date)).tz_localize(None),
                "open": float(p.open),
                "high": float(p.high),
                "low": float(p.low),
                "close": float(p.close),
                "volume": float(p.volume),
                "openinterest": 0.0,
            }
        )

    df = pd.DataFrame(rows).dropna(subset=["datetime"]).sort_values("datetime")
    if start:
        df = df[df["datetime"] >= pd.to_datetime(start)]
    if end:
        df = df[df["datetime"] <= pd.to_datetime(end)]
    if df.empty:
        raise ValueError(f"No price data for {symbol} in selected period (start={start}, end={end}). "
                        f"Available range: {df.index.min()} to {df.index.max()}" if len(df) > 0 else 
                        f"No price data for {symbol} in selected period (start={start}, end={end}).")

    df = df.set_index("datetime")
    return df


if bt is not None:
    class GraphSignalStrategy(bt.Strategy):  # type: ignore[misc]
        params = (
            ("graph", None),
            ("symbol", "AAPL"),
            ("min_confidence", 0.0),
            ("rebalance_threshold", 0.02),
            ("thread_prefix", "bt"),
            ("printlog", True),
            ("audit_path", None),
        )

        def __init__(self):
            if self.p.graph is None:
                raise ValueError("graph instance is required")
            self.order = None
            self.last_signal: Optional[Dict[str, Any]] = None
            self.daily_states: List[Dict[str, Any]] = []  # 记录每日状态，用于流式返回
            # 创建事件循环用于异步操作
            import asyncio
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

        def log(self, txt: str) -> None:
            if self.p.printlog:
                dt = self.datas[0].datetime.date(0).isoformat()
                print(f"[{dt}] {txt}")

        def notify_order(self, order):
            if order.status in [order.Submitted, order.Accepted]:
                return
            if order.status in [order.Completed]:
                side = "BUY" if order.isbuy() else "SELL"
                self.log(
                    f"{side} EXECUTED price={order.executed.price:.2f} "
                    f"size={order.executed.size:.0f} value={order.executed.value:.2f} "
                    f"comm={order.executed.comm:.2f}"
                )
            elif order.status in [order.Canceled, order.Margin, order.Rejected]:
                self.log(f"ORDER FAILED status={order.getstatusname()}")
            self.order = None

        def next(self):
            import time
            
            if self.order:
                return

            date_str = self.datas[0].datetime.date(0).isoformat()
            # 获取完整的 OHLCV 数据
            open_price = float(self.datas[0].open[0])
            high_price = float(self.datas[0].high[0])
            low_price = float(self.datas[0].low[0])
            close_price = float(self.datas[0].close[0])
            volume = float(self.datas[0].volume[0])
            cash = float(self.broker.getcash())
            portfolio_value = float(self.broker.getvalue())
            position = self.getposition(self.datas[0])
            position_size = int(position.size)
            avg_cost = float(position.price) if position_size != 0 else 0.0
            position_value = position_size * close_price
            current_pct = (position_value / portfolio_value) if portfolio_value > 0 else 0.0

            graph_input = build_graph_input(
                symbol=self.p.symbol,
                date_str=date_str,
                close_price=close_price,
                cash=cash,
                portfolio_value=portfolio_value,
                shares=position_size,
                avg_cost=avg_cost,
            )
            self.log(
                f"BAR state cash={cash:.2f} value={portfolio_value:.2f} "
                f"shares={position_size} avg_cost={avg_cost:.2f} close={close_price:.2f}"
            )
            
            # 添加延迟避免API限流
            time.sleep(0.5)
            
            # 设置回测日期环境变量，确保 Agent 知道"今天"是哪一天
            os.environ["FINGENT_SIMULATED_DATE"] = date_str
            
            thread_id = f"{self.p.thread_prefix}_{self.p.symbol}_{date_str}"
            # 使用事件循环运行异步方法
            graph_result = self._loop.run_until_complete(
                self.p.graph.run(graph_input, thread_id=thread_id)
            )

            signal = extract_graph_signal(
                graph_result=graph_result,
                current_position_pct=current_pct,
                symbol_fallback=self.p.symbol,
            )
            self.last_signal = asdict(signal)
            # 构建日志信息，包含策略原因
            log_msg = f"SIGNAL vote={signal.vote} confidence={signal.confidence:.1f} target={signal.target_position_pct:.2f}"
            if signal.reason:
                # 限制原因长度，避免日志过长
                reason_short = signal.reason[:100].replace('\n', ' ')
                log_msg += f" | reason={reason_short}"
            self.log(log_msg)

            # 记录每日状态（用于流式API）
            daily_state = {
                "date": date_str,
                "cash": cash,
                "portfolio_value": portfolio_value,
                "position_size": position_size,
                "avg_cost": avg_cost,
                "open_price": open_price,
                "high_price": high_price,
                "low_price": low_price,
                "close_price": close_price,
                "volume": volume,
                "signal": self.last_signal,
            }
            self.daily_states.append(daily_state)
            
            # 如果有回调函数，调用它（用于流式API实时推送）
            if hasattr(self, 'on_daily_update') and callable(self.on_daily_update):
                try:
                    self.on_daily_update(daily_state)
                except Exception as e:
                    self.log(f"回调错误: {e}")

            if self.p.audit_path:
                os.makedirs(os.path.dirname(self.p.audit_path) or ".", exist_ok=True)
                with open(self.p.audit_path, "a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {
                                "date": date_str,
                                "symbol": self.p.symbol,
                                "state": {
                                    "cash": cash,
                                    "portfolio_value": portfolio_value,
                                    "position_size": position_size,
                                    "avg_cost": avg_cost,
                                    "open_price": open_price,
                                    "high_price": high_price,
                                    "low_price": low_price,
                                    "close_price": close_price,
                                    "volume": volume,
                                    "current_position_pct": current_pct,
                                },
                                "graph_input": graph_input,
                                "graph_result": graph_result,
                                "signal": self.last_signal,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

            if signal.confidence < float(self.p.min_confidence):
                reason_short = signal.reason[:80].replace('\n', ' ') if signal.reason else ''
                self.log(
                    f"SKIP signal vote={signal.vote} confidence={signal.confidence:.1f} "
                    f"target={signal.target_position_pct:.2f} (below min_confidence)"
                    f" | reason={reason_short}"
                )
                return

            target_pct = float(signal.target_position_pct)
            diff = abs(target_pct - current_pct)
            if diff < float(self.p.rebalance_threshold):
                reason_short = signal.reason[:80].replace('\n', ' ') if signal.reason else ''
                self.log(
                    f"HOLD(no rebalance) vote={signal.vote} current={current_pct:.2f} "
                    f"target={target_pct:.2f} diff={diff:.2f}"
                    f" | reason={reason_short}"
                )
                return

            reason_short = signal.reason[:80].replace('\n', ' ') if signal.reason else ''
            self.log(
                f"REBALANCE vote={signal.vote} confidence={signal.confidence:.1f} "
                f"current={current_pct:.2f} target={target_pct:.2f}"
                f" | reason={reason_short}"
            )
            self.order = self.order_target_percent(data=self.datas[0], target=target_pct)
else:
    class GraphSignalStrategy:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            _require_backtrader()


def run_backtest(
    fin_graph,
    data: pd.DataFrame,
    symbol: str = "AAPL",
    initial_cash: float = 10000.0,
    commission: float = 0.001,
    slippage_perc: float = 0.0005,
    min_confidence: float = 0.0,
    rebalance_threshold: float = 0.02,
    printlog: bool = True,
    audit_path: Optional[str] = None,
) -> Dict[str, Any]:
    _require_backtrader()
    if data.empty:
        raise ValueError("Backtest data is empty.")

    cerebro = bt.Cerebro()
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=commission)
    if slippage_perc > 0:
        cerebro.broker.set_slippage_perc(slippage_perc)

    feed = bt.feeds.PandasData(dataname=data)
    cerebro.adddata(feed, name=symbol)
    cerebro.addstrategy(
        GraphSignalStrategy,
        graph=fin_graph,
        symbol=symbol,
        min_confidence=min_confidence,
        rebalance_threshold=rebalance_threshold,
        printlog=printlog,
        audit_path=audit_path,
    )

    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")

    start_value = float(cerebro.broker.getvalue())
    result = cerebro.run()
    strategy = result[0]
    end_value = float(cerebro.broker.getvalue())

    # 提取并格式化分析指标
    def safe_float(val, default=0.0):
        try:
            return float(val) if val is not None else default
        except (TypeError, ValueError):
            return default
    
    # 收益分析
    returns_analysis = strategy.analyzers.returns.get_analysis()
    # 回撤分析
    drawdown_analysis = strategy.analyzers.drawdown.get_analysis()
    # 夏普比率分析
    sharpe_analysis = strategy.analyzers.sharpe.get_analysis()
    # 交易分析
    trades_analysis = strategy.analyzers.trades.get_analysis()
    
    # 构建标准评估指标
    metrics = {
        "symbol": symbol,
        "start_value": round(start_value, 2),
        "end_value": round(end_value, 2),
        "pnl": round(end_value - start_value, 2),
        "total_return_pct": round(((end_value / start_value) - 1.0) * 100.0, 2) if start_value else 0.0,
        "annual_return_pct": round(safe_float(returns_analysis.get("rnorm100")), 2),
        "max_drawdown_pct": round(safe_float(drawdown_analysis.get("max", {}).get("drawdown")), 2),
        "sharpe_ratio": round(safe_float(sharpe_analysis.get("sharperatio")), 3),
        "volatility_ann_pct": round(safe_float(returns_analysis.get("rnorm100", 0)) / safe_float(sharpe_analysis.get("sharperatio", 1), 1), 2) if safe_float(sharpe_analysis.get("sharperatio")) > 0 else None,
    }
    
    # 交易统计
    total_trades = 0
    win_trades = 0
    loss_trades = 0
    win_rate = 0.0
    
    if trades_analysis and "total" in trades_analysis and "closed" in trades_analysis["total"]:
        total_trades = int(trades_analysis["total"]["closed"])
        if total_trades > 0:
            if "won" in trades_analysis and "total" in trades_analysis["won"]:
                win_trades = int(trades_analysis["won"]["total"])
            if "lost" in trades_analysis and "total" in trades_analysis["lost"]:
                loss_trades = int(trades_analysis["lost"]["total"])
            win_rate = round((win_trades / total_trades) * 100.0, 2) if total_trades > 0 else 0.0
    
    metrics["total_trades"] = total_trades
    metrics["win_trades"] = win_trades
    metrics["loss_trades"] = loss_trades
    metrics["win_rate_pct"] = win_rate
    
    return {
        **metrics,
        "last_signal": strategy.last_signal,
        "_raw_analyzers": {
            "drawdown": drawdown_analysis,
            "returns": returns_analysis,
            "sharpe": sharpe_analysis,
            "trades": trades_analysis,
        },
    }


def run_backtest_from_symbol(
    fin_graph,
    symbol: str = "AAPL",
    start: Optional[str] = None,
    end: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    data = load_price_dataframe(symbol=symbol, start=start, end=end)
    return run_backtest(fin_graph=fin_graph, data=data, symbol=symbol, **kwargs)
