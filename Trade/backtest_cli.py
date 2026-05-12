import json
import sys
import argparse
import os

sys.path.insert(0, ".")

from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI

# 加载环境变量
load_dotenv()

# 导入 FinGent 组件
from LLM.preprocess import Preprocessor
from LLM.router import Router
from LLM.agent import TECHNICAL_NERD, Morefit
from LLM.graph import FinGraph
from LLM.unified_stock_tools import (
    get_stock_price,
    get_stock_basic_info,
    get_stock_company_info,
    get_stock_financial_report_links,
    get_stock_financial_statements,
)

from Trade.runner import run_backtest_from_symbol


def create_fin_graph(temperature: float = 0.5):
    """创建并配置 FinGraph 实例"""
    api_key = os.getenv("QIANWEN_API_KEY")
    if not api_key:
        raise ValueError("请设置 QIANWEN_API_KEY 环境变量")

    # 使用内存数据库
    checkpointer = MemorySaver()
    model = ChatOpenAI(
        api_key=api_key,
        base_url=os.getenv(
            "MODEL_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
        ),
        model="qwen-max",
        temperature=temperature,
    )

    preprocessor = Preprocessor(model, checkpointer)
    router = Router()

    tech_agent = TECHNICAL_NERD(
        model=model,
        tools=[get_stock_price, get_stock_basic_info],
        checkpointer=checkpointer,
    )

    morefit_agent = Morefit(
        model=model,
        tools=[
            get_stock_company_info,
            get_stock_financial_report_links,
            get_stock_financial_statements,
            get_stock_price,
            get_stock_basic_info,
        ],
        checkpointer=checkpointer,
    )

    fin_graph = FinGraph(
        preprocessor=preprocessor,
        router=router,
        agent={"TECHNICAL_NERD": tech_agent, "Morefit": morefit_agent},
        checkpointer=checkpointer,
    )

    return fin_graph


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run FinGraph + backtrader backtest from CLI."
    )
    parser.add_argument("--symbol", default="GOOGL", help="Stock symbol, e.g. NVDA")
    parser.add_argument("--start", default="2025-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD")
    parser.add_argument(
        "--initial-cash", type=float, default=10000.0, help="Initial cash"
    )
    parser.add_argument(
        "--commission", type=float, default=0.001, help="Commission rate"
    )
    parser.add_argument(
        "--slippage", type=float, default=0.0005, help="Slippage percent"
    )
    parser.add_argument(
        "--min-confidence", type=float, default=0.0, help="Min signal confidence"
    )
    parser.add_argument(
        "--rebalance-threshold",
        type=float,
        default=0.02,
        help="Minimum target/current weight gap to rebalance",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable strategy logs",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="LLM temperature for backtest reproducibility (recommended 0.0)",
    )
    parser.add_argument(
        "--audit-path",
        default="backtest_audit.jsonl",
        help="Path to JSONL audit log for each bar",
    )
    return parser.parse_args()


def print_metrics_report(result: dict) -> None:
    """打印格式化的回测评估报告"""
    print("\n" + "=" * 60)
    print(f"📊 回测结果报告 - {result.get('symbol', 'N/A')}")
    print("=" * 60)

    print("\n【收益指标】")
    print(f"  初始资金:        ${result.get('start_value', 0):,.2f}")
    print(f"  最终资金:        ${result.get('end_value', 0):,.2f}")
    print(f"  盈亏金额:        ${result.get('pnl', 0):,.2f}")
    print(f"  总收益率:        {result.get('total_return_pct', 0):.2f}%")
    print(f"  年化收益率:      {result.get('annual_return_pct', 0):.2f}%")

    print("\n【风险指标】")
    print(f"  最大回撤:        {result.get('max_drawdown_pct', 0):.2f}%")
    print(f"  夏普比率:        {result.get('sharpe_ratio', 0):.3f}")
    vol = result.get("volatility_ann_pct")
    if vol is not None:
        print(f"  年化波动率:      {vol:.2f}%")

    print("\n【交易统计】")
    print(f"  总交易次数:      {result.get('total_trades', 0)}")
    print(f"  盈利次数:        {result.get('win_trades', 0)}")
    print(f"  亏损次数:        {result.get('loss_trades', 0)}")
    print(f"  胜率:            {result.get('win_rate_pct', 0):.2f}%")

    last_signal = result.get("last_signal")
    if last_signal:
        print("\n【最后交易信号】")
        print(f"  决策:            {last_signal.get('vote', 'N/A')}")
        print(f"  置信度:          {last_signal.get('confidence', 0):.1f}")
        print(
            f"  目标仓位:        {last_signal.get('target_position_pct', 0) * 100:.1f}%"
        )

    print("\n" + "=" * 60)


def main() -> int:
    args = parse_args()

    try:
        os.environ["FINGENT_BACKTEST_STRICT"] = "1"
        fin_graph = create_fin_graph(temperature=args.temperature)
        import asyncio

        result = asyncio.run(
            run_backtest_from_symbol(
                fin_graph=fin_graph,
                symbol=args.symbol,
                start=args.start,
                end=args.end,
                initial_cash=args.initial_cash,
                commission=args.commission,
                slippage_perc=args.slippage,
                min_confidence=args.min_confidence,
                rebalance_threshold=args.rebalance_threshold,
                printlog=(not args.quiet),
                audit_path=args.audit_path,
            )
        )
    except Exception as exc:
        print(f"Backtest failed: {exc}")
        return 1

    print_metrics_report(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
