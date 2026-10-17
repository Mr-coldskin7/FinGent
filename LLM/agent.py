# a class to build analysis agents to analyze stock
# I will build two main agents
# ===========================================================================================================
#
# First agent: TECHNICAL_NERD
# This agent possesses reverse-thinking capabilities and primarily focuses on the strategic interplay between
# institutional players (market manipulators) and retail investors as reflected in technical indicators.
# He has a clear understanding of the following principles:
# Institutional players, with their substantial capital, generally do not want retail investors to profit.
# Typically, before initiating a price rally, they deliberately create market volatility to force retail investors
# to exit their positions, after which they drive the price upward.
# Once the rally is underway and technical indicators show stable and positive momentum,
# they trap retail investors into holding losing positions.
# He never chases high prices or panics into selling during dips.
# Instead, he acts with boldness yet meticulous caution,
# demonstrating strong psychological resilience and a comprehensive, big-picture perspective.
# This agent is focused on the emotional aspects of stock market behavior.
#
# ===========================================================================================================
#
# Second agent: Morefit
# This agent's strategy focuses on stock fundamentals and the companies themselves.
# Like the early Warren Buffett, he seeks out undervalued stocks and boldly accumulates them.
# His core approach emphasizes the intrinsic value of businesses over market speculation.
# He pays close attention to corporate financial statements, analyzes the narratives behind those reports,
# and then-by understanding the company's operations-breaks its business down into smaller,
# relatable "mini-businesses" to estimate what the stock's fair price should be.
# Through this method, he identifies potentially undvalued stocks.
#
# ===========================================================================================================
try:
    import LLM.base as base
except ImportError:
    import base as base
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
import os
from langchain.tools import tool

# ────────────────────────────────────────────────────────────────────────
# 每个 Agent 的输出 JSON schema 已随 system_prompt 一起迁移至技能文件:
#   LLM/skills/technical-nerd/SKILL.md         (VOTE schema)
#   LLM/skills/risk-manager/SKILL.md           (RISK_VOTE schema)
#   LLM/skills/sentiment-analyzer/SKILL.md     (SENTIMENT_VOTE schema)
# 提示词/skills/tools 分离，修改输出格式请改对应 SKILL.md 而不是这里。
# ────────────────────────────────────────────────────────────────────────


class TECHNICAL_NERD(base.Agent):
    def format_prompt(self, stock: str, user_input: str) -> str:
        return f"请分析股票 {stock} 的技术面情况。用户原始问题：{user_input}"

    def __init__(
        self,
        model,
        tools=None,
        checkpointer=None,
        simulated_date=None,
        memory_manager=None,
        user_id=None,
    ):


        # 身份/分析协议/JSON输出格式已迁移至技能文件: LLM/skills/technical-nerd/SKILL.md
        super().__init__(
            model,
            tools or [],
            "",
            checkpointer,
            simulated_date,
            memory_manager=memory_manager,
            user_id=user_id,
            skill_name="technical-nerd",
        )


class Morefit(base.Agent):
    def format_prompt(self, stock: str, user_input: str) -> str:
        return (
            f"请分析股票：{stock}\n\n"
            f"要求：\n"
            f"1. 只分析这一只股票，不要查询其他股票\n"
            f"2. 用户问题：{user_input}\n"
            f"3. 使用工具获取数据后，输出纯JSON格式分析结果"
        )

    def __init__(
        self,
        model,
        tools=None,
        checkpointer=None,
        simulated_date=None,
        memory_manager=None,
        user_id=None,
    ):


        # 身份/分析协议/JSON输出格式已迁移至技能文件: LLM/skills/morefit/SKILL.md
        super().__init__(
            model,
            tools or [],
            "",
            checkpointer,
            simulated_date,
            memory_manager=memory_manager,
            user_id=user_id,
            skill_name="morefit",
        )


class RiskManager(base.Agent):
    """
    Risk Manager Agent - 风险与仓位控制专家

    核心职责：
    1. 量化评估单只股票和组合的风险暴露
    2. 基于凯利公式和固定比例法计算最优仓位
    3. 监控流动性风险、波动率、最大回撤
    4. 提供止损建议和仓位调整方案

    投资哲学：
    - 先求不败，再求胜。活下来比赚得多更重要。
    - 任何单笔交易的亏损不得超过组合净值的2%。
    - 仓位大小应该与确定性成反比，与风险成正比。
    - 高波动 ≠ 高收益，高波动 = 高不确定性 = 小仓位。
    """

    def __init__(
        self,
        model,
        tools=None,
        checkpointer=None,
        simulated_date=None,
        memory_manager=None,
        user_id=None,
    ):

        # 身份/分析协议/JSON输出格式已迁移至技能文件: LLM/skills/risk-manager/SKILL.md
        super().__init__(
            model,
            tools or [],
            "",
            checkpointer,
            simulated_date,
            memory_manager=memory_manager,
            user_id=user_id,
            skill_name="risk-manager",
        )


class SentimentAnalyzer(base.Agent):
    """
    Sentiment Analyzer Agent - 舆情分析专家

    核心职责：
    1. 监控和分析股票相关的新闻舆情
    2. 评估市场整体情绪和行业情绪
    3. 识别情绪转折点和市场极端情绪
    4. 提供基于舆情的中短期交易信号

    投资哲学：
    - 市场情绪是短期价格波动的最大驱动力。
    - 当所有人都在欢呼时，要保持警惕；当所有人都在恐慌时，要保持冷静。
    - 新闻情绪往往领先价格变化 1-3 天。
    - 极端情绪（过度乐观/悲观）是反向交易的信号。
    """

    def __init__(
        self,
        model,
        tools=None,
        checkpointer=None,
        simulated_date=None,
        memory_manager=None,
        user_id=None,
    ):

        # 身份/分析协议/JSON输出格式已迁移至技能文件: LLM/skills/sentiment-analyzer/SKILL.md
        super().__init__(
            model,
            tools or [],
            "",
            checkpointer,
            simulated_date,
            memory_manager=memory_manager,
            user_id=user_id,
            skill_name="sentiment-analyzer",
        )
