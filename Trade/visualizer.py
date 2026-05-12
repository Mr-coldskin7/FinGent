"""
回测结果可视化模块
支持：
1. 命令行生成回测图表
2. 标记买入/卖出点
3. 显示资金曲线
4. 交互式Web图表数据输出
"""

import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class TradeSignal:
    """交易信号"""

    date: str
    vote: str  # BUY, SELL, HOLD, STRONG_BUY, STRONG_SELL
    confidence: float
    target_position_pct: float
    price: float


@dataclass
class BacktestTrade:
    """回测交易记录"""

    date: str
    type: str  # 'buy' 或 'sell'
    price: float
    size: int
    value: float
    commission: float
    pnl: Optional[float] = None  # 卖出时的盈亏


@dataclass
class DailyRecord:
    """每日记录"""

    date: str
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: float
    portfolio_value: float
    cash: float
    position_size: int
    signal: Optional[Dict] = None


def parse_backtest_audit(audit_path: str = "Trade/backtest_audit.jsonl") -> List[Dict]:
    """解析回测审计日志"""
    records = []
    try:
        with open(audit_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
    except FileNotFoundError:
        print(f"审计文件不存在: {audit_path}")
        return []
    return records


def extract_trades_and_signals(records: List[Dict]) -> tuple:
    """从审计记录中提取交易点和信号"""
    daily_records = []
    trades = []

    prev_position = 0

    for record in records:
        date = record.get("date", "")
        state = record.get("state", {})
        signal = record.get("signal", {})

        # 获取收盘价（旧数据兼容）
        close_price = state.get("close_price", 0)

        # 兼容旧数据：如果没有OHLCV，使用收盘价填充
        open_price = state.get("open_price", close_price)
        high_price = state.get("high_price", close_price)
        low_price = state.get("low_price", close_price)
        volume = state.get("volume", 0)

        # 每日记录 - 包含完整 OHLCV
        daily_records.append(
            DailyRecord(
                date=date,
                open_price=open_price,
                high_price=high_price,
                low_price=low_price,
                close_price=close_price,
                volume=volume,
                portfolio_value=state.get("portfolio_value", 0),
                cash=state.get("cash", 0),
                position_size=state.get("position_size", 0),
                signal=signal,
            )
        )

        # 检测交易（仓位变化）
        current_position = state.get("position_size", 0)
        if current_position != prev_position:
            trade_type = "buy" if current_position > prev_position else "sell"
            size = abs(current_position - prev_position)
            price = state.get("close_price", 0)

            trade = BacktestTrade(
                date=date,
                type=trade_type,
                price=price,
                size=size,
                value=size * price,
                commission=0,  # 可从audit中解析
            )
            trades.append(trade)

        prev_position = current_position

    return daily_records, trades


def plot_backtest_matplotlib(
    daily_records: List[DailyRecord],
    trades: List[BacktestTrade],
    symbol: str = "STOCK",
    save_path: Optional[str] = None,
    show_plot: bool = True,
):
    """
    使用 Matplotlib 绘制回测结果
    包含：价格曲线、买卖点标记、资金曲线
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from datetime import datetime
    except ImportError:
        print("请先安装 matplotlib: pip install matplotlib")
        return None

    # 准备数据
    dates = [datetime.strptime(r.date, "%Y-%m-%d") for r in daily_records]
    prices = [r.close_price for r in daily_records]
    portfolio_values = [r.portfolio_value for r in daily_records]

    # 分离买卖交易
    buy_trades = [t for t in trades if t.type == "buy"]
    sell_trades = [t for t in trades if t.type == "sell"]

    # 创建图表
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [2, 1]}
    )

    # ========== 上图：价格曲线 + 买卖点 ==========
    ax1.plot(dates, prices, label="Close Price", color="#2563eb", linewidth=1.5)

    # 标记买入点（绿色向上箭头）
    for trade in buy_trades:
        trade_date = datetime.strptime(trade.date, "%Y-%m-%d")
        if trade_date in dates:
            idx = dates.index(trade_date)
            ax1.scatter(
                trade_date,
                trade.price,
                color="#22c55e",
                marker="^",
                s=150,
                zorder=5,
                edgecolors="white",
                linewidths=1,
            )
            ax1.annotate(
                f"BUY\n{trade.size}@{trade.price:.1f}",
                xy=(trade_date, trade.price),
                xytext=(0, 15),
                textcoords="offset points",
                ha="center",
                fontsize=7,
                color="#22c55e",
                fontweight="bold",
            )

    # 标记卖出点（红色向下箭头）
    for trade in sell_trades:
        trade_date = datetime.strptime(trade.date, "%Y-%m-%d")
        if trade_date in dates:
            idx = dates.index(trade_date)
            ax1.scatter(
                trade_date,
                trade.price,
                color="#ef4444",
                marker="v",
                s=150,
                zorder=5,
                edgecolors="white",
                linewidths=1,
            )
            ax1.annotate(
                f"SELL\n{trade.size}@{trade.price:.1f}",
                xy=(trade_date, trade.price),
                xytext=(0, -25),
                textcoords="offset points",
                ha="center",
                fontsize=7,
                color="#ef4444",
                fontweight="bold",
            )

    ax1.set_title(
        f"{symbol} Backtest - Price & Signals", fontsize=14, fontweight="bold"
    )
    ax1.set_ylabel("Price ($)", fontsize=11)
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax1.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

    # ========== 下图：资金曲线 ==========
    initial_value = portfolio_values[0] if portfolio_values else 10000
    ax2.fill_between(dates, portfolio_values, alpha=0.3, color="#8b5cf6")
    ax2.plot(
        dates, portfolio_values, label="Portfolio Value", color="#8b5cf6", linewidth=2
    )
    ax2.axhline(
        y=initial_value,
        color="gray",
        linestyle="--",
        alpha=0.5,
        label=f"Initial: ${initial_value:,.0f}",
    )

    # 计算收益率
    final_value = portfolio_values[-1] if portfolio_values else initial_value
    return_pct = ((final_value / initial_value) - 1) * 100
    color = "#22c55e" if return_pct >= 0 else "#ef4444"

    ax2.set_title(
        f"Portfolio Value | Return: {return_pct:+.2f}%",
        fontsize=12,
        color=color,
        fontweight="bold",
    )
    ax2.set_ylabel("Value ($)", fontsize=11)
    ax2.set_xlabel("Date", fontsize=11)
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax2.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

    # 添加统计信息
    stats_text = f"""
    Total Trades: {len(trades)}
    Buy: {len(buy_trades)} | Sell: {len(sell_trades)}
    Final Value: ${final_value:,.2f}
    """
    fig.text(
        0.02,
        0.02,
        stats_text,
        fontsize=9,
        verticalalignment="bottom",
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(
            save_path, dpi=150, bbox_inches="tight", facecolor="white", edgecolor="none"
        )
        print(f"图表已保存: {save_path}")

    if show_plot:
        plt.show()

    return fig


def generate_chart_data(
    daily_records: List[DailyRecord], trades: List[BacktestTrade], symbol: str = "STOCK"
) -> Dict[str, Any]:
    """
    生成前端图表所需的数据格式（包含K线数据）
    """
    # K线数据 (OHLCV)
    candles = [
        {
            "time": r.date,
            "open": round(r.open_price, 2),
            "high": round(r.high_price, 2),
            "low": round(r.low_price, 2),
            "close": round(r.close_price, 2),
            "volume": round(r.volume, 0),
        }
        for r in daily_records
    ]

    # 交易点数据 (使用 time 字段匹配 TradingView 格式)
    trade_markers = [
        {
            "time": t.date,
            "type": t.type,
            "price": t.price,
            "size": t.size,
            "value": t.value,
        }
        for t in trades
    ]

    # 统计信息
    initial_value = daily_records[0].portfolio_value if daily_records else 10000
    final_value = daily_records[-1].portfolio_value if daily_records else initial_value
    return_pct = ((final_value / initial_value) - 1) * 100 if initial_value else 0

    return {
        "symbol": symbol,
        "candles": candles,
        "trade_markers": trade_markers,
        "statistics": {
            "initial_value": initial_value,
            "final_value": final_value,
            "return_pct": round(return_pct, 2),
            "total_trades": len(trades),
            "buy_count": len([t for t in trades if t.type == "buy"]),
            "sell_count": len([t for t in trades if t.type == "sell"]),
        },
    }


def visualize_backtest_cli(
    audit_path: str = "Trade/backtest_audit.jsonl",
    symbol: str = "STOCK",
    save_path: Optional[str] = None,
):
    """
    命令行入口：可视化回测结果

    用法:
        python -m Trade.visualizer
        python -m Trade.visualizer --symbol NVDA --save nvda_backtest.png
    """
    print("=" * 60)
    print("FinGent 回测可视化工具")
    print("=" * 60)

    # 解析数据
    records = parse_backtest_audit(audit_path)
    if not records:
        print("没有回测数据可供可视化")
        return

    print(f"加载了 {len(records)} 条回测记录")

    # 提取交易和信号
    daily_records, trades = extract_trades_and_signals(records)
    print(
        f"识别到 {len(trades)} 笔交易 ({len([t for t in trades if t.type=='buy'])} 买 / {len([t for t in trades if t.type=='sell'])} 卖)"
    )

    # 生成图表
    if save_path is None:
        save_path = (
            f"backtest_chart_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        )

    plot_backtest_matplotlib(daily_records, trades, symbol, save_path)

    print("=" * 60)
    print("可视化完成!")
    print(f"图表保存: {save_path}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FinGent 回测可视化工具")
    parser.add_argument(
        "--audit", default="Trade/backtest_audit.jsonl", help="回测审计文件路径"
    )
    parser.add_argument("--symbol", default="STOCK", help="股票代码")
    parser.add_argument("--save", default=None, help="保存图表的路径")
    parser.add_argument("--no-show", action="store_true", help="不显示图表，只保存")

    args = parser.parse_args()

    visualize_backtest_cli(args.audit, args.symbol, args.save)
