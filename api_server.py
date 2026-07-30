from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import uuid
from datetime import datetime
from dotenv import load_dotenv
from sse_starlette.sse import EventSourceResponse
import asyncio
from typing import Optional
import queue
from queue import Queue as _Queue
import pandas as pd
from dataclasses import dataclass, field

load_dotenv()
from langgraph.checkpoint.redis.aio import AsyncRedisSaver

# ========== 统一配置 ==========
from config import get_settings, get_model_config, update_model_config, reset_model_config, ModelConfig, MODEL_PRESETS

settings = get_settings()

# ========== 初始化 Graph ==========
from langchain_openai import ChatOpenAI
from LLM.preprocess import Preprocessor
from LLM.router import Router
from LLM.agent import TECHNICAL_NERD, Morefit, RiskManager, SentimentAnalyzer
from LLM.graph import FinGraph
from LLM.unified_stock_tools import *
from LLM.risk_tools import RISK_TOOLS
from LLM.sentiment_tools import SENTIMENT_TOOLS
from Data.memory import get_memory_manager

# Redis 持久化 - 使用上下文管理器
_redis_ctx = AsyncRedisSaver.from_conn_string(settings.redis_url)
checkpointer = asyncio.run(_redis_ctx.__aenter__())

# L3 长期记忆管理器
memory_manager = get_memory_manager()


def build_fin_graph(model_config: ModelConfig) -> FinGraph:
    """工厂函数：根据模型配置创建 FinGraph 实例"""
    llm = ChatOpenAI(
        api_key=model_config.api_key,
        base_url=model_config.base_url,
        model=model_config.model_name,
        temperature=model_config.temperature,
    )
    return FinGraph(
        preprocessor=Preprocessor(llm, checkpointer),
        router=Router(),
        agent={
            "TECHNICAL_NERD": TECHNICAL_NERD(
                llm,
                [
                    get_stock_price,
                    get_stock_basic_info,
                    search_financial_knowledge,
                    get_financial_indicator_explanation,
                    bocha_search,
                ],
                checkpointer,
                memory_manager=memory_manager,
            ),
            "Morefit": Morefit(
                llm,
                [
                    get_stock_company_info,
                    get_stock_financial_report_links,
                    get_stock_financial_statements,
                    get_stock_price,
                    get_stock_basic_info,
                    search_financial_knowledge,
                    get_financial_indicator_explanation,
                    bocha_search,
                ],
                checkpointer,
                memory_manager=memory_manager,
            ),
            "RiskManager": RiskManager(
                llm, RISK_TOOLS, checkpointer, memory_manager=memory_manager
            ),
            "SentimentAnalyzer": SentimentAnalyzer(
                llm, SENTIMENT_TOOLS, checkpointer, memory_manager=memory_manager
            ),
        },
        checkpointer=checkpointer,
        memory_manager=memory_manager,
    )


# 全局 FinGraph 实例 + 并发锁
fin_graph = build_fin_graph(get_model_config())
_graph_lock = asyncio.Lock()


# 注册关闭钩子（程序退出时清理）
import atexit


@atexit.register
def cleanup():
    try:
        asyncio.run(_redis_ctx.__aexit__(None, None, None))
    except Exception:
        pass


# ========== API ==========
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    user_input: str
    thread_id: Optional[str] = None
    user_id: Optional[str] = "anonymous"


class FeedbackRequest(BaseModel):
    session_id: str
    stock_symbol: str
    agent_name: Optional[str] = None  # 如果针对特定 agent，否则影响全局
    feedback: str  # agree | disagree | correction
    rule_text: Optional[str] = None  # 用户纠正内容，correction 时必填
    user_id: Optional[str] = "anonymous"


class ReviewRequest(BaseModel):
    session_id: Optional[str] = None
    stock_symbol: Optional[str] = None
    user_id: Optional[str] = "anonymous"


@app.post("/api/v1/feedback")
async def feedback(req: FeedbackRequest):
    """
    用户反馈接口 — Self-Improve 被动触发入口
    - agree: 提升对应 agent 权重
    - disagree/correction: 降低权重，并记录 user_rules
    """
    try:
        mm = memory_manager
        # 1. 更新 analysis_history 中的 feedback（找最近一条匹配记录）
        # 2. 更新 agent_stats 权重
        feedback_lower = req.feedback.lower()
        if req.session_id and req.stock_symbol:
            await mm.record_feedback(req.session_id, req.stock_symbol, feedback_lower)

        if feedback_lower == "agree":
            if req.agent_name:
                new_weight = await mm.adjust_weight(
                    req.agent_name, 0.05, req.user_id
                )
                await mm.record_agent_outcome(req.agent_name, "agree", user_id=req.user_id)
                return {
                    "success": True,
                    "action": "weight_increased",
                    "agent": req.agent_name,
                    "new_weight": new_weight,
                }
            return {"success": True, "message": "全局同意已记录（未指定 Agent）"}

        elif feedback_lower in ("disagree", "correction"):
            if req.agent_name:
                new_weight = await mm.adjust_weight(
                    req.agent_name, -0.10, req.user_id
                )
                await mm.record_agent_outcome(
                    req.agent_name, feedback_lower, user_id=req.user_id
                )
            if req.rule_text:
                rule = await mm.add_or_update_rule(
                    rule_text=req.rule_text,
                    agent_name=req.agent_name or "ALL",
                    user_id=req.user_id,
                    source="explicit_feedback",
                )
                return {
                    "success": True,
                    "action": "rule_recorded",
                    "agent": req.agent_name,
                    "new_weight": new_weight if req.agent_name else None,
                    "rule_id": rule.id,
                    "trigger_count": rule.trigger_count,
                }
            return {
                "success": True,
                "action": "weight_decreased",
                "agent": req.agent_name,
                "new_weight": new_weight if req.agent_name else None,
            }
        else:
            return {"success": False, "error": f"未知反馈类型: {req.feedback}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/api/v1/chat")
async def chat(req: ChatRequest):
    """直接透传给 Graph.run()"""
    thread_id = (
        req.thread_id or f"chat_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"
    )

    async with _graph_lock:
        current_graph = fin_graph
    result = await current_graph.run(req.user_input, thread_id, user_id=req.user_id)

    print(f"\n{'='*60}")
    print(f"DEBUG: Graph result keys: {list(result.keys())}")
    print(f"{'='*60}\n")

    response = {
        "success": True,
        "thread_id": thread_id,
        "input": result.get("input"),
        "parse_result": result.get("parse_result"),
        "result": result.get("result"),
        "status": result.get("status"),
    }

    if result.get("tool_chain"):
        response["tool_chain"] = result["tool_chain"]
    if result.get("agent_name"):
        response["agent_name"] = result["agent_name"]
    if result.get("stock"):
        response["stock"] = result["stock"]
    if result.get("all_tool_chains"):
        response["all_tool_chains"] = result["all_tool_chains"]
    if result.get("final_decision"):
        response["final_decision"] = result["final_decision"]
    if result.get("detailed_analysis"):
        response["detailed_analysis"] = result["detailed_analysis"]
    if result.get("user_id"):
        response["user_id"] = result["user_id"]

    return response


# 热门美股列表
US_HOT_STOCKS = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "TSLA",
    "NVDA",
    "META",
    "NFLX",
    "AMD",
    "INTC",
    "BABA",
    "JD",
    "PDD",
    "TSM",
    "COIN",
    "CRM",
    "ADBE",
    "UBER",
    "LYFT",
    "PLTR",
]


# ========== 模型管理 API ==========


class ModelConfigRequest(BaseModel):
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model_name: Optional[str] = None
    temperature: Optional[float] = None
    preset: Optional[str] = None  # 预设提供商名称


class ModelTestRequest(BaseModel):
    api_key: str
    base_url: str
    model_name: str


@app.get("/api/v1/model/config")
async def get_model_config_api():
    """获取当前模型配置（api_key 脱敏）"""
    config = get_model_config()
    return {
        "success": True,
        "config": config.to_dict(mask_key=True),
        "presets": MODEL_PRESETS,
    }


@app.put("/api/v1/model/config")
async def update_model_config_api(req: ModelConfigRequest):
    """更新模型配置并重建 FinGraph"""
    global fin_graph

    current = get_model_config()

    # 应用预设
    if req.preset and req.preset in MODEL_PRESETS:
        preset = MODEL_PRESETS[req.preset]
        new_config = ModelConfig(
            api_key=req.api_key or current.api_key,
            base_url=preset["base_url"],
            model_name=preset["model_name"],
            temperature=req.temperature if req.temperature is not None else current.temperature,
        )
    else:
        new_config = ModelConfig(
            api_key=req.api_key if req.api_key is not None else current.api_key,
            base_url=req.base_url if req.base_url is not None else current.base_url,
            model_name=req.model_name if req.model_name is not None else current.model_name,
            temperature=req.temperature if req.temperature is not None else current.temperature,
        )

    # 验证：api_key 不能为空
    if not new_config.api_key:
        return {"success": False, "error": "API Key 不能为空"}

    # 重建 FinGraph
    try:
        async with _graph_lock:
            fin_graph = build_fin_graph(new_config)
        update_model_config(new_config)
        return {
            "success": True,
            "message": "模型配置已更新",
            "config": new_config.to_dict(mask_key=True),
        }
    except Exception as e:
        return {"success": False, "error": f"重建模型失败: {str(e)}"}


@app.post("/api/v1/model/test")
async def test_model_connection(req: ModelTestRequest):
    """测试模型连接（发送一个简单请求验证 api_key + base_url）"""
    try:
        test_llm = ChatOpenAI(
            api_key=req.api_key,
            base_url=req.base_url,
            model=req.model_name,
            temperature=0.1,
        )
        # 发一个最简单的请求测试连通性
        from langchain_core.messages import HumanMessage

        response = await test_llm.ainvoke([HumanMessage(content="Hi")])
        return {
            "success": True,
            "message": "连接成功",
            "model_response_preview": response.content[:100],
        }
    except Exception as e:
        return {"success": False, "error": f"连接失败: {str(e)}"}


@app.post("/api/v1/model/reset")
async def reset_model_config_api():
    """重置为 .env 默认配置"""
    global fin_graph

    try:
        default_config = reset_model_config()
        async with _graph_lock:
            fin_graph = build_fin_graph(default_config)
        return {
            "success": True,
            "message": "已恢复默认配置",
            "config": default_config.to_dict(mask_key=True),
        }
    except Exception as e:
        return {"success": False, "error": f"重置失败: {str(e)}"}


@app.get("/api/v1/market")
async def get_market(market: str = "zh", limit: int = 50):
    """
    获取实时行情数据
    - market=zh: A股实时行情（按涨跌幅绝对值排序）
    - market=us: 美股热门股票行情
    """
    if market == "zh":
        return await _get_zh_market(limit)
    elif market == "us":
        return await _get_us_market()
    else:
        return {"success": False, "error": "不支持的市场类型，请使用 zh 或 us"}


@app.post("/api/v1/review")
async def review(req: ReviewRequest):
    if not req.session_id and not req.stock_symbol:
        return {"success": False, "error": "需要 session_id 或 stock_symbol 来进行复盘。"}

    latest_record = None
    if req.session_id:
        latest_record = await memory_manager.get_analysis_record(
            req.session_id, req.stock_symbol
        )
    if not latest_record and req.stock_symbol:
        recent = await memory_manager.get_recent_analyses(
            req.user_id, req.stock_symbol, limit=1
        )
        latest_record = recent[0] if recent else None

    if not latest_record:
        return {"success": False, "error": "未找到对应的分析记录。"}

    history = await memory_manager.get_recent_analyses(
        req.user_id, latest_record.stock_symbol, limit=5
    )
    agent_weights = await memory_manager.get_agent_weights(
        user_id=req.user_id, agent_names=list(latest_record.agent_votes.keys())
    )
    inconsistency = await memory_manager.find_inconsistencies(
        user_id=req.user_id,
        stock_symbol=latest_record.stock_symbol,
        current_decision=latest_record.final_decision,
    )

    return {
        "success": True,
        "review": {
            "stock_symbol": latest_record.stock_symbol,
            "current_decision": latest_record.final_decision,
            "reasoning_summary": latest_record.reasoning_summary,
            "user_feedback": latest_record.user_feedback,
            "agent_votes": latest_record.agent_votes,
            "agent_weights": agent_weights,
            "recent_history": [
                {
                    "created_at": r.created_at,
                    "final_decision": r.final_decision,
                    "user_feedback": r.user_feedback,
                    "query": r.query,
                }
                for r in history
            ],
            "inconsistency_warning": inconsistency,
        },
    }


async def _get_zh_market(limit: int):
    """获取A股实时行情"""
    try:
        from Data.providers import zh_stock

        df = await zh_stock.recent_stock_list()

        if df is None or df.empty:
            return {"success": False, "error": "无法获取A股行情数据"}

        # 按涨跌幅绝对值排序，取前N条
        df = df.copy()
        df["涨跌幅_绝对值"] = df["涨跌幅"].abs()
        df = df.sort_values("涨跌幅_绝对值", ascending=False).head(limit)

        stocks = []
        for _, row in df.iterrows():
            stocks.append(
                {
                    "symbol": str(row.get("代码", "")),
                    "name": str(row.get("名称", "")),
                    "price": (
                        float(row.get("最新价", 0))
                        if pd.notna(row.get("最新价"))
                        else 0
                    ),
                    "change": (
                        float(row.get("涨跌额", 0))
                        if pd.notna(row.get("涨跌额"))
                        else 0
                    ),
                    "changePercent": (
                        float(row.get("涨跌幅", 0))
                        if pd.notna(row.get("涨跌幅"))
                        else 0
                    ),
                    "volume": (
                        int(row.get("成交量", 0)) if pd.notna(row.get("成交量")) else 0
                    ),
                    "turnover": (
                        float(row.get("成交额", 0))
                        if pd.notna(row.get("成交额"))
                        else 0
                    ),
                    "high": (
                        float(row.get("最高", 0)) if pd.notna(row.get("最高")) else 0
                    ),
                    "low": (
                        float(row.get("最低", 0)) if pd.notna(row.get("最低")) else 0
                    ),
                    "open": (
                        float(row.get("今开", 0)) if pd.notna(row.get("今开")) else 0
                    ),
                    "prevClose": (
                        float(row.get("昨收", 0)) if pd.notna(row.get("昨收")) else 0
                    ),
                }
            )

        return {"success": True, "market": "zh", "count": len(stocks), "stocks": stocks}
    except Exception as e:
        return {"success": False, "error": f"获取A股行情失败: {str(e)}"}


async def _get_us_market():
    """获取美股热门股票行情（通过 Tiingo API）"""
    try:
        import httpx
        from datetime import datetime, timedelta

        token = settings.tiingo_api_key
        if not token:
            return {"success": False, "error": "TIINGO_API_KEY 未配置"}

        # 获取最近 5 天的数据用于计算涨跌幅
        start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

        stocks = []
        async with httpx.AsyncClient(timeout=30.0) as client:
            for symbol in US_HOT_STOCKS:
                try:
                    url = (
                        f"https://api.tiingo.com/tiingo/daily/{symbol}/prices"
                        f"?startDate={start_date}&token={token}"
                    )
                    r = await client.get(url)
                    data = r.json()

                    if not isinstance(data, list) or len(data) < 2:
                        continue

                    latest = data[-1]
                    prev = data[-2]

                    price = float(latest.get("close", 0))
                    prev_close = float(prev.get("close", 0))
                    change = price - prev_close
                    change_percent = (change / prev_close * 100) if prev_close else 0
                    volume = int(latest.get("volume", 0)) if latest.get("volume") else 0

                    stocks.append(
                        {
                            "symbol": symbol,
                            "name": symbol,  # Tiingo 不返回名称，用 symbol 代替
                            "price": round(price, 2),
                            "change": round(change, 2),
                            "changePercent": round(change_percent, 2),
                            "volume": volume,
                            "dayHigh": float(latest.get("high", 0)),
                            "dayLow": float(latest.get("low", 0)),
                        }
                    )
                except Exception:
                    continue

        # 按涨跌幅绝对值排序
        stocks.sort(key=lambda x: abs(x["changePercent"]), reverse=True)

        return {"success": True, "market": "us", "count": len(stocks), "stocks": stocks}
    except Exception as e:
        return {"success": False, "error": f"获取美股行情失败: {str(e)}"}


class BacktestRequest(BaseModel):
    symbol: str = "GOOGL"
    start: str = "2024-01-01"
    end: Optional[str] = None
    initial_cash: float = 10000.0
    commission: float = 0.001
    slippage: float = 0.0005
    min_confidence: float = 0.0
    rebalance_threshold: float = 0.02
    quiet: bool = False
    temperature: float = 0.0
    audit_path: str = "Trade/backtest_audit.jsonl"
    session_id: Optional[str] = None  # 回测会话 ID，用于并发隔离
    interval: str = "daily"  # 数据频率：daily / weekly / monthly / annually


@dataclass
class BacktestSession:
    """单个回测会话的状态容器，支持多用户并发回测"""

    session_id: str
    data_queue: _Queue = field(default_factory=lambda: _Queue(maxsize=1000))
    cancelled: bool = False
    done: bool = False
    error: Optional[str] = None


# 回测会话管理：按 session_id 隔离
backtest_sessions: dict[str, BacktestSession] = {}


def get_or_create_session(session_id: Optional[str] = None) -> tuple[str, BacktestSession]:
    """获取或创建回测会话，限制并发数"""
    if session_id and session_id in backtest_sessions:
        existing = backtest_sessions[session_id]
        if existing.done:
            del backtest_sessions[session_id]
        else:
            return session_id, existing

    # 限制并发回测数
    active_count = sum(1 for s in backtest_sessions.values() if not s.done)
    if active_count >= settings.backtest_max_concurrent:
        # 清理已完成的会话
        done_keys = [k for k, v in backtest_sessions.items() if v.done]
        for k in done_keys:
            del backtest_sessions[k]

    sid = session_id or f"bt_{uuid.uuid4().hex[:8]}"
    session = BacktestSession(session_id=sid)
    backtest_sessions[sid] = session
    return sid, session


@app.post("/api/v1/backtest")
async def backtest(req: BacktestRequest):
    """执行回测任务的API接口"""
    try:
        from Trade.runner import run_backtest_from_symbol
        import asyncio

        os.environ["FINGENT_BACKTEST_STRICT"] = "1"

        mc = get_model_config()
        temp_model = ChatOpenAI(
            api_key=mc.api_key,
            base_url=mc.base_url,
            model=mc.model_name,
            temperature=req.temperature,
        )
        _redis_ctx = AsyncRedisSaver.from_conn_string(settings.redis_url)
        temp_checkpointer = await _redis_ctx.__aenter__()

        temp_graph = FinGraph(
            preprocessor=Preprocessor(temp_model, temp_checkpointer),
            router=Router(),
            agent={
                "TECHNICAL_NERD": TECHNICAL_NERD(
                    temp_model,
                    [get_stock_price, get_stock_basic_info],
                    temp_checkpointer,
                ),
                "Morefit": Morefit(
                    temp_model,
                    [
                        get_stock_company_info,
                        get_stock_financial_report_links,
                        get_stock_financial_statements,
                        get_stock_price,
                        get_stock_basic_info,
                    ],
                    temp_checkpointer,
                ),
            },
            checkpointer=temp_checkpointer,
        )

        try:
            result = await run_backtest_from_symbol(
                fin_graph=temp_graph,
                symbol=req.symbol,
                start=req.start,
                end=req.end,
                resample_freq=req.interval,
                initial_cash=req.initial_cash,
                commission=req.commission,
                slippage_perc=req.slippage,
                min_confidence=req.min_confidence,
                rebalance_threshold=req.rebalance_threshold,
                printlog=(not req.quiet),
                audit_path=req.audit_path,
            )
        finally:
            await _redis_ctx.__aexit__(None, None, None)

        return {
            "success": True,
            "result": result,
            "symbol": req.symbol,
            "period": {"start": req.start, "end": req.end},
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def create_strategy_with_callback(
    StrategyClass, data_queue, cancel_check_fn=None, **kwargs
):
    """创建一个带有回调函数的策略类"""

    class CallbackStrategy(StrategyClass):
        def __init__(self):
            super().__init__()
            self.data_queue = data_queue
            self.cancel_check_fn = cancel_check_fn

        def on_daily_update(self, state):
            """回调函数：将每日状态推送到队列"""
            try:
                self.data_queue.put(state, block=False)
            except queue.Full:
                pass  # 队列满了，丢弃旧数据

        def next(self):
            # 检查是否被取消
            if self.cancel_check_fn and self.cancel_check_fn():
                self.log("回测已取消，正在停止...")
                self.env.runstop()
                return
            # 调用父类的 next
            super().next()

    # 复制原策略的参数定义
    CallbackStrategy.params = StrategyClass.params
    return CallbackStrategy


@app.post("/api/v1/backtest-stream")
async def backtest_stream(req: BacktestRequest):
    """流式回测API - 逐日返回结果（支持并发，按 session_id 隔离）

    SSE 事件格式:
    - event: start - 回测开始
    - event: daily_update - 每日更新
    - event: final_result - 最终结果
    - event: error - 错误信息
    - event: cancelled - 回测被取消
    - event: ping - 保持连接
    """
    import json
    import threading
    from Trade.runner import GraphSignalStrategy, load_price_dataframe
    import backtrader as bt

    # 创建或获取回测会话（并发隔离）
    sid, session = get_or_create_session(req.session_id)

    result_holder = {
        "strategy": None,
        "start_value": 0,
        "end_value": 0,
        "analyzers": {},
        "error": None,
        "done": False,
    }

    def run_backtest():
        """在后台线程中运行回测"""
        try:
            os.environ["FINGENT_BACKTEST_STRICT"] = "1"

            print(f"[StreamBacktest:{sid}] 加载 {req.symbol} 数据...")
            data = asyncio.run(
                load_price_dataframe(symbol=req.symbol, start=req.start, end=req.end, resample_freq=req.interval)
            )
            print(f"[StreamBacktest:{sid}] 加载完成: {len(data)} 条")

            if session.cancelled:
                print(f"[StreamBacktest:{sid}] 回测在启动前被取消")
                session.done = True
                return

            mc = get_model_config()
            temp_model = ChatOpenAI(
                api_key=mc.api_key,
                base_url=mc.base_url,
                model=mc.model_name,
                temperature=req.temperature,
            )
            _redis_ctx = AsyncRedisSaver.from_conn_string(settings.redis_url)
            temp_checkpointer = asyncio.run(_redis_ctx.__aenter__())

            temp_graph = FinGraph(
                preprocessor=Preprocessor(temp_model, temp_checkpointer),
                router=Router(),
                agent={
                    "TECHNICAL_NERD": TECHNICAL_NERD(
                        temp_model,
                        [get_stock_price, get_stock_basic_info],
                        temp_checkpointer,
                    ),
                    "Morefit": Morefit(
                        temp_model,
                        [
                            get_stock_company_info,
                            get_stock_financial_report_links,
                            get_stock_financial_statements,
                            get_stock_price,
                            get_stock_basic_info,
                        ],
                        temp_checkpointer,
                    ),
                },
                checkpointer=temp_checkpointer,
            )

            cerebro = bt.Cerebro()
            cerebro.broker.setcash(req.initial_cash)
            cerebro.broker.setcommission(commission=req.commission)
            if req.slippage > 0:
                cerebro.broker.set_slippage_perc(req.slippage)

            feed = bt.feeds.PandasData(dataname=data)
            cerebro.adddata(feed, name=req.symbol)

            cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
            cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
            cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
            cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")

            CallbackStrategy = create_strategy_with_callback(
                GraphSignalStrategy,
                session.data_queue,
                cancel_check_fn=lambda: session.cancelled,
            )

            cerebro.addstrategy(
                CallbackStrategy,
                graph=temp_graph,
                symbol=req.symbol,
                min_confidence=req.min_confidence,
                rebalance_threshold=req.rebalance_threshold,
                printlog=True,
                audit_path=req.audit_path,
            )

            result_holder["start_value"] = float(cerebro.broker.getvalue())
            print(f"[StreamBacktest:{sid}] 开始运行回测...")

            if session.cancelled:
                print(f"[StreamBacktest:{sid}] 回测在运行前被取消")
                session.done = True
                return

            results = cerebro.run()

            if session.cancelled:
                print(f"[StreamBacktest:{sid}] 回测被取消")
            else:
                print(f"[StreamBacktest:{sid}] 回测完成!")

            if results:
                strategy = results[0]
                result_holder["strategy"] = strategy
                result_holder["end_value"] = float(cerebro.broker.getvalue())
                try:
                    result_holder["analyzers"] = {
                        "drawdown": strategy.analyzers.drawdown.get_analysis(),
                        "returns": strategy.analyzers.returns.get_analysis(),
                        "sharpe": strategy.analyzers.sharpe.get_analysis(),
                        "trades": strategy.analyzers.trades.get_analysis(),
                    }
                except Exception:
                    pass

            result_holder["done"] = True

        except Exception as e:
            print(f"[StreamBacktest:{sid}] 错误: {e}")
            import traceback

            traceback.print_exc()
            result_holder["error"] = str(e)
            result_holder["done"] = True
        finally:
            session.done = True

    async def event_generator():
        """生成器：从队列读取数据并yield"""

        thread = threading.Thread(target=run_backtest, daemon=True)
        thread.start()

        total_days_estimate = 252
        if req.end:
            try:
                start_date = pd.to_datetime(req.start)
                end_date = pd.to_datetime(req.end)
                total_days_estimate = len(pd.bdate_range(start_date, end_date))
            except Exception:
                pass

        yield {
            "event": "start",
            "data": json.dumps(
                {
                    "session_id": sid,
                    "symbol": req.symbol,
                    "message": "回测开始",
                    "period": {"start": req.start, "end": req.end},
                    "total_days": total_days_estimate,
                },
                ensure_ascii=False,
            ),
        }

        total_days = 0
        max_wait = settings.backtest_timeout
        waited = 0
        last_data_time = asyncio.get_event_loop().time()

        while not result_holder["done"] or not session.data_queue.empty():
            if session.cancelled:
                print(f"[StreamBacktest:{sid}] 检测到取消信号，停止事件生成")
                yield {
                    "event": "cancelled",
                    "data": json.dumps(
                        {"message": "回测已取消", "total_days": total_days},
                        ensure_ascii=False,
                    ),
                }
                return

            try:
                state = session.data_queue.get(block=False)
                total_days += 1

                daily_result = {
                    "date": str(state["date"]),
                    "cash": float(state["cash"]),
                    "portfolio_value": float(state["portfolio_value"]),
                    "position_size": (
                        int(state["position_size"])
                        if state["position_size"] is not None
                        else 0
                    ),
                    "avg_cost": (
                        float(state["avg_cost"]) if state["avg_cost"] is not None else 0
                    ),
                    "open_price": float(state.get("open_price", state["close_price"])),
                    "high_price": float(state.get("high_price", state["close_price"])),
                    "low_price": float(state.get("low_price", state["close_price"])),
                    "close_price": float(state["close_price"]),
                    "volume": float(state.get("volume", 0)),
                    "day_number": total_days,
                }

                if "signal" in state and state["signal"]:
                    daily_result["signal"] = {
                        "vote": state["signal"].get("vote", ""),
                        "confidence": float(state["signal"].get("confidence", 0)),
                        "target_position_pct": float(
                            state["signal"].get("target_position_pct", 0)
                        ),
                        "reason": state["signal"].get("reason", ""),
                    }

                yield {
                    "event": "daily_update",
                    "data": json.dumps(daily_result, ensure_ascii=False, default=str),
                }

                last_data_time = asyncio.get_event_loop().time()
                await asyncio.sleep(0.01)

            except queue.Empty:
                if not result_holder["done"]:
                    await asyncio.sleep(0.1)
                    waited += 0.1

                    time_since_last_data = (
                        asyncio.get_event_loop().time() - last_data_time
                    )
                    if time_since_last_data > 300:
                        yield {
                            "event": "ping",
                            "data": json.dumps(
                                {"status": "waiting", "elapsed": time_since_last_data},
                                ensure_ascii=False,
                            ),
                        }
                        last_data_time = asyncio.get_event_loop().time()

                    if waited > max_wait:
                        yield {
                            "event": "error",
                            "data": json.dumps(
                                {"error": f"回测执行超时（{max_wait/60:.0f}分钟）"},
                                ensure_ascii=False,
                            ),
                        }
                        return

        thread.join(timeout=5)

        if session.cancelled:
            print(f"[StreamBacktest:{sid}] 回测已被取消，跳过最终结果")
        elif result_holder["error"]:
            yield {
                "event": "error",
                "data": json.dumps(
                    {"error": result_holder["error"]}, ensure_ascii=False
                ),
            }
        elif result_holder["strategy"]:
            final_result = {
                "session_id": sid,
                "symbol": req.symbol,
                "start_value": result_holder["start_value"],
                "end_value": result_holder["end_value"],
                "pnl": result_holder["end_value"] - result_holder["start_value"],
                "return_pct": (
                    ((result_holder["end_value"] / result_holder["start_value"]) - 1.0)
                    * 100.0
                    if result_holder["start_value"]
                    else 0.0
                ),
                "last_signal": result_holder["strategy"].last_signal,
                "total_days": total_days,
                "analyzers": result_holder["analyzers"],
            }

            yield {
                "event": "final_result",
                "data": json.dumps(final_result, ensure_ascii=False, default=str),
            }

    return EventSourceResponse(
        event_generator(),
        ping=15,
        ping_message_factory=lambda: {"event": "ping", "data": "{}"},
    )


@app.post("/api/v1/backtest-cancel")
async def cancel_backtest(session_id: Optional[str] = None):
    """
    取消正在进行的回测（按 session_id 隔离）
    """
    if session_id and session_id in backtest_sessions:
        backtest_sessions[session_id].cancelled = True
        print(f"[Backtest:{session_id}] 收到取消请求")
        return {"success": True, "message": f"回测 {session_id} 取消信号已发送"}
    # 兜底：取消所有活跃会话
    cancelled = []
    for sid, s in backtest_sessions.items():
        if not s.done:
            s.cancelled = True
            cancelled.append(sid)
    if cancelled:
        print(f"[Backtest] 批量取消: {cancelled}")
        return {"success": True, "message": f"已取消 {len(cancelled)} 个回测会话", "cancelled_sessions": cancelled}
    return {"success": False, "message": "没有活跃的回测会话"}


@app.get("/api/v1/backtest-chart")
async def get_backtest_chart(
    audit_path: str = "Trade/backtest_audit.jsonl", symbol: str = "STOCK"
):
    """
    获取回测图表数据，用于前端可视化

    参数:
        audit_path: 回测审计文件路径
        symbol: 股票代码

    返回:
        {
            "symbol": "NVDA",
            "candles": [{"time": "2024-01-01", "open": 100, "high": 105, "low": 99, "close": 102, "volume": 10000}],
            "trade_markers": [{"time": "2024-01-05", "type": "buy", "price": 102.0, "size": 10}],
            "statistics": {...}
        }
    """
    import os

    print(f"[ChartAPI] 请求图表数据: symbol={symbol}, audit_path={audit_path}")

    try:
        from Trade.visualizer import (
            parse_backtest_audit,
            extract_trades_and_signals,
            generate_chart_data,
        )

        # 检查文件是否存在
        if not os.path.exists(audit_path):
            # 尝试其他路径
            alt_paths = [
                f"Trade/{audit_path}",
                audit_path.replace("Trade/", ""),
                os.path.join("Trade", "backtest_audit.jsonl"),
                "backtest_audit.jsonl",
            ]
            for alt_path in alt_paths:
                if os.path.exists(alt_path):
                    audit_path = alt_path
                    print(f"[ChartAPI] 使用替代路径: {audit_path}")
                    break

        print(
            f"[ChartAPI] 最终审计文件路径: {audit_path}, 存在: {os.path.exists(audit_path)}"
        )

        # 解析回测数据
        records = parse_backtest_audit(audit_path)
        print(f"[ChartAPI] 解析到 {len(records)} 条记录")

        if not records:
            return {
                "success": False,
                "error": f"没有找到回测数据，请先运行回测 (路径: {audit_path})",
            }

        # 提取交易信息
        daily_records, trades = extract_trades_and_signals(records)
        print(f"[ChartAPI] 提取到 {len(daily_records)} 日记录, {len(trades)} 笔交易")

        # 生成图表数据
        chart_data = generate_chart_data(daily_records, trades, symbol)
        print(f"[ChartAPI] 生成图表数据: {len(chart_data.get('candles', []))} 根K线")

        return {"success": True, "data": chart_data}

    except Exception as e:
        import traceback

        print(f"[ChartAPI] 错误: {e}")
        traceback.print_exc()
        return {"success": False, "error": str(e)}


@app.get("/health")
async def health():
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=settings.api_host, port=settings.api_port)
