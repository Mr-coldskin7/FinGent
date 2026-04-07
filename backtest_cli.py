#!/usr/bin/env python3
"""
FinGent 回测命令行工具
支持：运行回测 + 终端可视化

用法:
    python backtest_cli.py --symbol NVDA --start 2024-01-01 --end 2024-06-01
    python backtest_cli.py --symbol AAPL --visualize  # 运行并显示图表
    python backtest_cli.py --visualize-only  # 仅显示已有回测结果
"""

import argparse
import os
import sys
from datetime import datetime

# 设置路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from dotenv import load_dotenv
load_dotenv()


def run_backtest_cli(args):
    """运行回测"""
    print("=" * 70)
    print(f"🚀 FinGent 回测 - {args.symbol}")
    print("=" * 70)
    
    # 导入依赖
    try:
        from langgraph.checkpoint.redis.aio import AsyncRedisSaver
        from langchain_community.chat_models import ChatTongyi
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
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        print("请确保在正确的环境中运行")
        return None
    
    # 检查 API Key
    api_key = os.getenv("QIANWEN_API_KEY")
    if not api_key:
        print("❌ 未设置 QIANWEN_API_KEY 环境变量")
        print("请先设置: set QIANWEN_API_KEY=your_api_key")
        return None
    
    # 初始化模型
    print("\n[1/4] 初始化模型...")
    model = ChatTongyi(api_key=api_key, temperature=args.temperature)
    # 使用 Redis 作为异步 Checkpoint 存储
    import asyncio
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    _redis_ctx = AsyncRedisSaver.from_conn_string(redis_url)
    checkpointer = asyncio.run(_redis_ctx.__aenter__())
    
    # 构建 Graph
    print("[2/4] 构建 FinGraph...")
    fin_graph = FinGraph(
        preprocessor=Preprocessor(model, checkpointer),
        router=Router(),
        agent={
            'TECHNICAL_NERD': TECHNICAL_NERD(model, [get_stock_price, get_stock_basic_info], checkpointer),
            'Morefit': Morefit(model, [get_stock_company_info, get_stock_financial_report_links, 
                                       get_stock_financial_statements, get_stock_price, get_stock_basic_info], 
                              checkpointer)
        },
        checkpointer=checkpointer
    )
    
    # 运行回测
    print(f"[3/4] 运行回测: {args.symbol}")
    print(f"      期间: {args.start} ~ {args.end or '至今'}")
    print(f"      初始资金: ${args.initial_cash:,.2f}")
    print(f"      手续费: {args.commission * 100:.2f}%")
    print(f"      滑点: {args.slippage * 100:.3f}%")
    print()
    
    start_time = datetime.now()
    
    try:
        result = run_backtest_from_symbol(
            fin_graph=fin_graph,
            symbol=args.symbol,
            start=args.start,
            end=args.end,
            initial_cash=args.initial_cash,
            commission=args.commission,
            slippage_perc=args.slippage,
            min_confidence=args.min_confidence,
            rebalance_threshold=args.rebalance_threshold,
            printlog=True,
            audit_path=args.audit_path
        )
        
        elapsed = (datetime.now() - start_time).total_seconds()
        
        print(f"\n[4/4] 回测完成! 耗时: {elapsed:.1f}秒")
        return result
        
    except Exception as e:
        print(f"\n❌ 回测失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def display_terminal_chart(audit_path: str, symbol: str):
    """在终端显示简单的ASCII图表"""
    try:
        from Trade.visualizer import parse_backtest_audit, extract_trades_and_signals
    except ImportError:
        print("请先安装 matplotlib: pip install matplotlib")
        return
    
    records = parse_backtest_audit(audit_path)
    if not records:
        print("没有回测数据可供显示")
        return
    
    daily_records, trades = extract_trades_and_signals(records)
    
    if not daily_records:
        print("没有每日记录数据")
        return
    
    # 简化的ASCII图表
    prices = [r.close_price for r in daily_records]
    portfolio_values = [r.portfolio_value for r in daily_records]
    
    min_price = min(prices)
    max_price = max(prices)
    price_range = max_price - min_price if max_price != min_price else 1
    
    min_val = min(portfolio_values)
    max_val = max(portfolio_values)
    val_range = max_val - min_val if max_val != min_val else 1
    
    # 显示统计
    initial = daily_records[0].portfolio_value
    final = daily_records[-1].portfolio_value
    return_pct = ((final / initial) - 1) * 100
    
    print("\n" + "=" * 70)
    print(f"📊 {symbol} 回测结果")
    print("=" * 70)
    print(f"  初始资金: ${initial:,.2f}")
    print(f"  最终资金: ${final:,.2f}")
    print(f"  收益率: {return_pct:+.2f}%")
    print(f"  交易次数: {len(trades)} ({len([t for t in trades if t.type == 'buy'])} 买 / {len([t for t in trades if t.type == 'sell'])} 卖)")
    print("=" * 70)
    
    # ASCII价格走势
    print("\n📈 价格走势 (简化):")
    chart_width = 50
    for i, record in enumerate(daily_records):
        if i % max(1, len(daily_records) // 10) == 0:  # 只显示约10个点
            price_bar_len = int((record.close_price - min_price) / price_range * chart_width)
            bar = "█" * price_bar_len + "░" * (chart_width - price_bar_len)
            marker = ""
            # 检查是否有交易
            for t in trades:
                if t.date == record.date:
                    marker = " [BUY ↑]" if t.type == "buy" else " [SELL ↓]"
            print(f"  {record.date} |{bar}| ${record.close_price:.2f}{marker}")
    
    # ASCII资金曲线
    print("\n💰 资金曲线:")
    for i, record in enumerate(daily_records):
        if i % max(1, len(daily_records) // 10) == 0:
            val_bar_len = int((record.portfolio_value - min_val) / val_range * chart_width)
            bar = "█" * val_bar_len + "░" * (chart_width - val_bar_len)
            print(f"  {record.date} |{bar}| ${record.portfolio_value:,.0f}")
    
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description='FinGent 回测命令行工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 运行回测并显示图表
  python backtest_cli.py --symbol NVDA --visualize
  
  # 只运行回测
  python backtest_cli.py --symbol AAPL --start 2024-01-01 --end 2024-06-01
  
  # 只显示已有回测结果的图表
  python backtest_cli.py --visualize-only --symbol NVDA
        """
    )
    
    parser.add_argument('--symbol', default='AAPL', help='股票代码 (默认: AAPL)')
    parser.add_argument('--start', default='2024-01-01', help='开始日期 (默认: 2024-01-01)')
    parser.add_argument('--end', default=None, help='结束日期 (默认: 至今)')
    parser.add_argument('--initial-cash', type=float, default=10000, help='初始资金 (默认: 10000)')
    parser.add_argument('--commission', type=float, default=0.001, help='手续费率 (默认: 0.001)')
    parser.add_argument('--slippage', type=float, default=0.0005, help='滑点 (默认: 0.0005)')
    parser.add_argument('--min-confidence', type=float, default=0.0, help='最小置信度 (默认: 0.0)')
    parser.add_argument('--rebalance-threshold', type=float, default=0.02, help='再平衡阈值 (默认: 0.02)')
    parser.add_argument('--temperature', type=float, default=0.0, help='模型温度 (默认: 0.0)')
    parser.add_argument('--audit-path', default='Trade/backtest_audit.jsonl', help='审计日志路径')
    parser.add_argument('-v', '--visualize', action='store_true', help='运行回测后显示图表')
    parser.add_argument('--visualize-only', action='store_true', help='仅显示已有回测结果的图表')
    parser.add_argument('--save-chart', default=None, help='保存图表路径')
    
    args = parser.parse_args()
    
    # 仅可视化模式
    if args.visualize_only:
        display_terminal_chart(args.audit_path, args.symbol)
        
        # 同时生成matplotlib图表
        try:
            from Trade.visualizer import visualize_backtest_cli
            visualize_backtest_cli(args.audit_path, args.symbol, args.save_chart)
        except Exception as e:
            print(f"Matplotlib 图表生成失败: {e}")
        return
    
    # 运行回测
    result = run_backtest_cli(args)
    
    if result:
        # 显示终端图表
        display_terminal_chart(args.audit_path, args.symbol)
        
        # 如果需要可视化，生成matplotlib图表
        if args.visualize:
            print("\n📊 生成可视化图表...")
            try:
                from Trade.visualizer import visualize_backtest_cli
                visualize_backtest_cli(args.audit_path, args.symbol, args.save_chart)
            except Exception as e:
                print(f"图表生成失败: {e}")
                print("请确保已安装 matplotlib: pip install matplotlib")


if __name__ == "__main__":
    main()
