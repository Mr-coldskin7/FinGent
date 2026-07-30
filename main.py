"""
FinGent - AI股票分析助手
主程序入口

用法:
    python main.py                    # 启动交互式对话
    python main.py --web             # 启动Web服务(如需要)
    python main.py "分析茅台走势"     # 单次查询模式
"""

import os
import sys
import asyncio
import argparse
from datetime import datetime
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 统一配置
from config import get_settings
settings = get_settings()

# 导入LLM模块
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from langchain_openai import ChatOpenAI

# 导入FinGent组件
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
from Data.memory import get_memory_manager
import atexit

# Redis checkpointer 上下文（全局，供清理使用）
_redis_ctx = None


@atexit.register
def cleanup():
    """程序退出时清理 Redis 连接"""
    if _redis_ctx is not None:
        try:
            asyncio.run(_redis_ctx.__aexit__(None, None, None))
        except Exception:
            pass


def get_model(temperature: float = 0.5):
    """获取配置的LLM模型"""
    if not settings.qianwen_api_key:
        raise ValueError("请设置 QIANWEN_API_KEY 环境变量")

    return ChatOpenAI(
        api_key=settings.qianwen_api_key,
        base_url=settings.model_base_url,
        model=settings.model_name,
        temperature=temperature,
    )


def create_fin_graph(temperature: float = 0.5):
    """
    创建并配置FinGraph实例

    Args:
        temperature: LLM温度参数，用于控制输出的随机性

    Returns:
        FinGraph: 配置好的图实例
    """
    # 初始化组件 - 使用 Redis 作为 checkpointer
    global _redis_ctx
    _redis_ctx = AsyncRedisSaver.from_conn_string(settings.redis_url)
    checkpointer = asyncio.run(_redis_ctx.__aenter__())
    model = get_model(temperature=temperature)
    memory_manager = get_memory_manager()

    preprocessor = Preprocessor(model, checkpointer)
    router = Router()

    # 创建Agent实例 - 使用统一股票工具，注入 L3 记忆管理器
    tech_agent = TECHNICAL_NERD(
        model=model,
        tools=[get_stock_price, get_stock_basic_info],
        checkpointer=checkpointer,
        memory_manager=memory_manager,
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
        memory_manager=memory_manager,
    )

    # 构建图
    fin_graph = FinGraph(
        preprocessor=preprocessor,
        router=router,
        agent={"TECHNICAL_NERD": tech_agent, "Morefit": morefit_agent},
        checkpointer=checkpointer,
        memory_manager=memory_manager,
    )

    return fin_graph


async def interactive_mode(fin_graph: FinGraph):
    """
    交互式对话模式

    支持多轮对话、澄清、线程管理
    """
    print("\n" + "=" * 60)
    print("🤖 FinGent - AI股票分析助手")
    print("=" * 60)
    print("\n支持美股和A股分析，输入股票代码或名称即可")
    print("命令:")
    print("  /help     - 显示帮助")
    print("  /reset    - 开始新对话")
    print("  /exit     - 退出")
    print("-" * 60)

    thread_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    while True:
        try:
            user_input = input("\n👤 你: ").strip()

            if not user_input:
                continue

            # 命令处理
            if user_input.lower() in ["/exit", "/quit", "exit", "quit"]:
                print("\n再见！👋")
                break

            elif user_input == "/help":
                print("\n📖 使用示例:")
                print("  • 分析AAPL的技术面")
                print("  • 茅台估值怎么样？")
                print("  •  Tesla值得买吗？")
                print("  • 600519财报分析")
                continue

            elif user_input == "/reset":
                thread_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                print("\n🔄 已开启新对话")
                continue

            # 执行分析
            result = await fin_graph.run(user_input, thread_id)

            # 处理需要澄清的情况
            if result.get("status") == "waiting_for_clarification":
                clarification = result.get("result", "请提供更多信息")
                print(f"\n🤖 助手: {clarification}")

                # 等待用户澄清
                clarification_input = input("\n👤 补充: ").strip()
                if clarification_input:
                    result = await fin_graph.resume(clarification_input, thread_id)
                    print(f"\n🤖 助手:\n{result.get('result', '分析完成')}")
            else:
                # 正常输出结果
                print(f"\n🤖 助手:\n{result.get('result', '无结果')}")

        except KeyboardInterrupt:
            print("\n\n再见！👋")
            break
        except Exception as e:
            print(f"\n❌ 错误: {e}")


async def single_query(fin_graph: FinGraph, query: str):
    """
    单次查询模式

    Args:
        fin_graph: FinGraph实例
        query: 用户查询
    """
    thread_id = f"single_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print(f"\n🔍 查询: {query}")
    print("-" * 60)

    result = await fin_graph.run(query, thread_id)

    if result.get("status") == "waiting_for_clarification":
        print(f"\n⚠️ 需要澄清: {result.get('result')}")
    else:
        print(f"\n📊 结果:\n{result.get('result', '无结果')}")

    return result


async def test_dual_agent_voting(fin_graph: FinGraph, symbol: str = "AAPL"):
    """
    测试双Agent投票模式 - 模拟回测场景

    展示 Technical_Nerd 和 Morefit 两个Agent的独立判断和最终投票结果
    """
    import json

    thread_id = f"voting_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print("\n" + "=" * 70)
    print(f"🗳️ 双Agent投票测试 - {symbol}")
    print("=" * 70)

    # 构造模拟回测的输入（包含账户状态）
    portfolio_context = {
        "cash": 10000.00,
        "portfolio_value": 10000.00,
        "holdings": [
            {"symbol": symbol, "shares": 0, "avg_cost": 0.0, "market_value": 0.0}
        ],
        "constraints": {"max_position_pct": 0.5, "trade_size": 10},
    }

    user_input = (
        f"You are making one-bar trading intent for backtest.\n"
        f"symbol={symbol}\n"
        f"date=2025-12-05\n"
        f"close_price=278.78\n"
        f"portfolio_context={json.dumps(portfolio_context, ensure_ascii=True)}\n"
        f"\n请分析并给出交易决策（BUY/HOLD/SELL）和目标仓位比例（target_position_pct）。"
    )

    print(f"\n📋 输入信息:")
    print(f"   股票: {symbol}")
    print(f"   价格: $278.78")
    print(f"   现金: $10,000")
    print(f"   持仓: 0 shares")

    print(f"\n⏳ 正在调用双Agent分析（可能需要30-60秒）...")
    print("-" * 70)

    result = await fin_graph.run(user_input, thread_id=thread_id)

    # 解析并展示投票结果
    print("\n" + "=" * 70)
    print("📊 投票结果分析")
    print("=" * 70)

    final_decision = result.get("final_decision", {})
    morefit_vote = result.get("morefit_vote", {})
    tech_vote = result.get("tech_vote", {})

    # Morefit (基本面)
    print(f"\n🏢 Morefit (基本面分析):")
    print(f"   投票: {morefit_vote.get('vote', 'N/A')}")
    print(f"   理由: {morefit_vote.get('reason', 'N/A')[:150]}...")
    if "target_position_pct" in morefit_vote:
        print(f"   目标仓位: {morefit_vote.get('target_position_pct')*100:.1f}%")

    # Technical_Nerd (技术面)
    print(f"\n📈 Technical_Nerd (技术面分析):")
    print(f"   投票: {tech_vote.get('vote', 'N/A')}")
    print(f"   理由: {tech_vote.get('reason', 'N/A')[:150]}...")
    if "target_position_pct" in tech_vote:
        print(f"   目标仓位: {tech_vote.get('target_position_pct')*100:.1f}%")

    # 最终结果
    print(f"\n" + "-" * 70)
    print(f"🏆 最终决策:")
    print(f"   投票: {final_decision.get('final_vote', 'N/A')}")
    print(f"   置信度: {final_decision.get('confidence', 'N/A')}%")
    print(f"   建议: {final_decision.get('suggestion', 'N/A')}")
    if "target_position_pct" in final_decision:
        print(f"   目标仓位: {final_decision.get('target_position_pct')*100:.1f}%")

    print(f"\n" + "=" * 70)

    # 返回完整结果供进一步检查
    return {
        "morefit": morefit_vote,
        "technical": tech_vote,
        "final": final_decision,
        "raw": result,
    }


async def main_async():
    """异步主入口函数"""
    parser = argparse.ArgumentParser(
        description="FinGent - AI股票分析助手",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py                          # 交互式模式
  python main.py "分析茅台的技术面"        # 单次查询
  python main.py --query "AAPL值得买吗"    # 显式查询参数
        """,
    )

    parser.add_argument("query", nargs="?", help="单次查询内容（不指定则进入交互模式）")

    parser.add_argument("--query", "-q", dest="explicit_query", help="显式指定查询内容")

    parser.add_argument(
        "--test-voting",
        "-t",
        dest="test_voting",
        metavar="SYMBOL",
        nargs="?",
        const="AAPL",
        help="测试双Agent投票模式 (默认: AAPL)",
    )

    args = parser.parse_args()

    # 确定查询内容
    query = args.explicit_query or args.query

    try:
        # 创建FinGraph
        print("🚀 正在初始化 FinGent...")
        fin_graph = create_fin_graph()
        print("✅ 初始化完成\n")

        if args.test_voting:
            # 双Agent投票测试模式
            await test_dual_agent_voting(fin_graph, args.test_voting)
        elif query:
            # 单次查询模式
            await single_query(fin_graph, query)
        else:
            # 交互式模式
            await interactive_mode(fin_graph)

    except ValueError as e:
        print(f"\n❌ 配置错误: {e}")
        print("请确保 .env 文件中设置了 QIANWEN_API_KEY")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 运行错误: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


def main():
    """同步入口，运行异步主函数"""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
