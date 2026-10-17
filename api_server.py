from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import json
import time
import uuid
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from sse_starlette.sse import EventSourceResponse
import asyncio
from typing import Any, Dict, List, Optional
from contextlib import asynccontextmanager
from typing import AsyncIterator
import queue
from queue import Queue as _Queue
import pandas as pd
from dataclasses import dataclass, field

load_dotenv()

# ========== 统一配置 ==========
from config import get_settings, get_model_config, update_model_config, reset_model_config, ModelConfig, MODEL_PRESETS
from api_auth import decide, validate_token

settings = get_settings()

# ========== 初始化 Graph ==========
from LLM.graph import FinGraph, assemble_fin_graph
from LLM.checkpointer import get_checkpointer, get_checkpointer_async
from LLM.llm import create_chat_model
from Data.memory import canonical_agent_name, get_memory_manager

# Checkpointer 可插拔：FINGENT_CHECKPOINTER=memory(默认)|redis
checkpointer, _cleanup_ckpt = get_checkpointer()

# L3 长期记忆管理器
memory_manager = get_memory_manager()


# 回测只用两个 Agent（速度优先，逐日 LLM 调用成本高）
BACKTEST_AGENT_NAMES = ("technical-nerd", "morefit")


def build_fin_graph(model_config: ModelConfig) -> FinGraph:
    """工厂函数：根据模型配置创建 FinGraph 实例（agent 全部由注册表装配）。

    新增 DIY agent = 丢 LLM/skills/<name>/SKILL.md 即自动注册并参与投票。
    """
    llm = create_chat_model(
        api_key=model_config.api_key,
        base_url=model_config.base_url,
        model_name=model_config.model_name,
        temperature=model_config.temperature,
    )
    return assemble_fin_graph(llm, checkpointer, memory_manager)


# 全局 FinGraph 实例 + 并发锁
fin_graph = build_fin_graph(get_model_config())
_graph_lock = asyncio.Lock()


# 注册关闭钩子（程序退出时清理）
import atexit


@atexit.register
def cleanup():
    try:
        _cleanup_ckpt()
    except Exception:
        pass


# ========== API ==========
@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """应用生命周期：启动时预热行情缓存（替代已弃用的 on_event）。"""
    await _prewarm_market_cache()
    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_methods=["*"],
    allow_headers=["Authorization", "Content-Type"],
)

# 鉴权：判定逻辑见 api_auth.py（默认拒绝；公开接口清单也在那里）
validate_token(settings.api_token)  # token 过短则拒绝启动
if not settings.api_token and settings.api_host not in ("127.0.0.1", "localhost", "::1"):
    print("⚠️  API 监听非本机地址，但未设置 FINGENT_API_TOKEN：受保护接口仅允许本机访问。")


@app.middleware("http")
async def api_access_control(request: Request, call_next):
    decision = decide(
        method=request.method,
        path=request.url.path,
        authorization=request.headers.get("authorization"),
        client_host=request.client.host if request.client else "",
        token=get_settings().api_token,
    )
    if not decision.allowed:
        return JSONResponse(
            {"success": False, "error": decision.message, "code": decision.code},
            status_code=decision.status,
            headers={"WWW-Authenticate": "Bearer"} if decision.status == 401 else None,
        )
    return await call_next(request)


class ChatRequest(BaseModel):
    user_input: str
    thread_id: Optional[str] = None
    user_id: Optional[str] = "anonymous"
    agents: Optional[List[str]] = None  # 显式指定参与分析的 agent（注册表名）；缺省=语义选择


class FeedbackRequest(BaseModel):
    session_id: str
    stock_symbol: str
    agent_name: Optional[str] = None  # 单 Agent 结果上的反馈
    agent_names: Optional[List[str]] = None  # 多 Agent 结果上用户勾选的 Agent（优先于 agent_name）
    feedback: str  # agree | disagree | correction
    rule_text: Optional[str] = None  # 用户纠正内容，correction 时必填
    user_id: Optional[str] = "anonymous"


class ReviewRequest(BaseModel):
    session_id: Optional[str] = None
    stock_symbol: Optional[str] = None
    user_id: Optional[str] = "anonymous"


def _direction(vote: str) -> str:
    """把投票归为方向：bullish / bearish / neutral。"""
    if vote in ("BUY", "STRONG_BUY"):
        return "bullish"
    if vote in ("SELL", "STRONG_SELL", "REDUCE"):
        return "bearish"
    return "neutral"


@app.post("/api/v1/feedback")
async def feedback(req: FeedbackRequest):
    """
    用户反馈接口（Self-Improve 被动入口）
    - 始终写入该分析记录的 user_feedback（历史）
    - 归因：agent_names（多 Agent 勾选）优先，其次 agent_name；都没有则不改任何权重
    - agree: 对应 Agent 的个人权重 +delta；disagree/correction: -delta，并可记录纠正规则
    权重口径与投票一致：有效权重 = 客观权重(global) × 该用户的个人调整。
    """
    try:
        mm = memory_manager
        user_id = req.user_id or "anonymous"
        feedback_lower = req.feedback.lower()
        if feedback_lower not in ("agree", "disagree", "correction"):
            return {"success": False, "error": f"未知反馈类型: {req.feedback}"}

        if req.session_id and req.stock_symbol:
            await mm.record_feedback(req.session_id, req.stock_symbol, feedback_lower)

        raw_names = [n for n in (req.agent_names or []) if n] or ([req.agent_name] if req.agent_name else [])
        names = list(dict.fromkeys(canonical_agent_name(n) for n in raw_names))  # 归一化 + 去重
        # 注意：record_agent_outcome 内部已按反馈调整 current_weight，
        # 不能再叠加 adjust_weight（同一 delta 会应用两次，review P1 已修）。
        weights: Dict[str, float] = {}
        for name in names:
            weights[name] = await mm.record_agent_outcome(name, feedback_lower, user_id=user_id)

        rule = None
        if feedback_lower in ("disagree", "correction") and req.rule_text:
            rule = await mm.add_or_update_rule(
                rule_text=req.rule_text,
                agent_name=(names[0] if len(names) == 1 else "ALL"),
                user_id=user_id,
                source="explicit_feedback",
            )

        if not names:
            message = "已记录反馈（未指定具体 Agent，权重未调整）"
            action = "feedback_recorded"
        elif feedback_lower == "agree":
            message = f"已提升 {'、'.join(names)} 的个人权重"
            action = "weight_increased"
        else:
            message = f"已降低 {'、'.join(names)} 的个人权重"
            action = "weight_decreased"
        if rule is not None:
            action = "rule_recorded"

        response: Dict[str, Any] = {
            "success": True,
            "action": action,
            "message": message,
            "agents": names,
            "weights": weights,
        }
        if len(names) == 1:
            response["agent"] = names[0]
            response["new_weight"] = weights[names[0]]
        if rule is not None:
            response["rule_id"] = rule.id
            response["trigger_count"] = rule.trigger_count
        return response
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/api/v1/review")
async def review(req: ReviewRequest):
    """历史决策复盘：当前决策、理由、投票拆解、结果验证、近期历史与方向反转提醒。"""
    try:
        mm = memory_manager
        stock = (req.stock_symbol or "").strip()
        if not stock:
            return {"success": False, "error": "缺少股票代码，无法复盘"}
        user_id = req.user_id or "anonymous"

        latest = None
        if req.session_id:
            latest = await mm.get_analysis_record(req.session_id, stock)
        if latest is None:
            found = await mm.get_recent_analyses(user_id, stock, limit=1)
            latest = found[0] if found else None
        if latest is None:
            return {"success": False, "error": f"没有找到 {stock} 的历史分析记录"}

        recent = await mm.get_recent_analyses(user_id, stock, limit=5)
        # 方向反转：最新一条 vs 它之前的一条（排除自身）
        inconsistency = await mm.find_inconsistencies(
            user_id, stock, latest.final_decision, exclude_id=latest.id
        )
        votes = latest.agent_votes or {}
        final_dir = _direction(latest.final_decision)
        dissenters = [a for a, v in votes.items() if _direction(v) != final_dir]
        outcomes = await mm.analysis_outcomes(latest.id)

        return {
            "success": True,
            "review": {
                "analysis_id": latest.id,
                "stock_symbol": stock,
                "created_at": latest.created_at,
                "current_decision": latest.final_decision,
                "reasoning_summary": latest.reasoning_summary,
                "user_feedback": latest.user_feedback,
                "agent_votes": votes,
                "dissenting_agents": dissenters,
                "outcomes": outcomes,
                "inconsistency_warning": inconsistency,
                "recent_history": [
                    {
                        "id": r.id,
                        "created_at": r.created_at,
                        "final_decision": r.final_decision,
                        "user_feedback": r.user_feedback,
                    }
                    for r in recent
                ],
            },
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


class RuleUpdate(BaseModel):
    user_id: str = "anonymous"
    active: bool


@app.get("/api/v1/memory/rules")
async def list_memory_rules(user_id: str = "anonymous", include_inactive: bool = True):
    """列出用户规则（含已禁用）。"""
    rules = await memory_manager.list_rules(user_id, include_inactive=include_inactive)
    return {"success": True, "rules": rules}


@app.patch("/api/v1/memory/rules/{rule_id}")
async def update_memory_rule(rule_id: int, req: RuleUpdate):
    """启用/禁用规则（只能操作自己的规则）。"""
    ok = await memory_manager.set_rule_active(rule_id, req.user_id, req.active)
    if not ok:
        return {"success": False, "error": "规则不存在或不属于该用户"}
    return {"success": True, "rule_id": rule_id, "active": req.active}


@app.delete("/api/v1/memory/rules/{rule_id}")
async def delete_memory_rule(rule_id: int, user_id: str = "anonymous"):
    """删除规则（只能删除自己的规则）。"""
    ok = await memory_manager.delete_rule(rule_id, user_id)
    if not ok:
        return {"success": False, "error": "规则不存在或不属于该用户"}
    return {"success": True, "rule_id": rule_id}


@app.get("/api/v1/memory/accuracy")
async def memory_accuracy(horizon: int = 20):
    """各 Agent 的前向命中率（结果验证产出）。"""
    return {"success": True, "horizon": horizon, "agents": await memory_manager.accuracy_report(horizon)}


def _build_chat_response(result: dict, thread_id: str) -> dict:
    """把 Graph.run 的返回值封装成前端协议（/chat 与 /chat-stream 共用）"""
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
    if result.get("selected_agents"):
        response["selected_agents"] = result["selected_agents"]
    if result.get("route_info"):
        response["route_info"] = result["route_info"]
    if result.get("user_id"):
        response["user_id"] = result["user_id"]
    return response


@app.post("/api/v1/chat")
async def chat(req: ChatRequest):
    """直接透传给 Graph.run()（带超时与异常兜底，任何错误都返回结构化 JSON）"""
    thread_id = (
        req.thread_id or f"chat_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"
    )

    try:
        async with _graph_lock:
            current_graph = fin_graph
        result = await asyncio.wait_for(
            current_graph.run(
                req.user_input, thread_id, user_id=req.user_id, agents=req.agents
            ),
            timeout=300,  # 5 分钟整体超时，避免 LLM 挂起拖死演示
        )
    except asyncio.TimeoutError:
        return {
            "success": False,
            "thread_id": thread_id,
            "error": "分析超时了（5分钟），请换个问题或稍后再试",
        }
    except Exception as e:
        import traceback

        traceback.print_exc()
        return {"success": False, "thread_id": thread_id, "error": f"分析出错: {e}"}

    return _build_chat_response(result, thread_id)


@app.post("/api/v1/chat-stream")
async def chat_stream(req: ChatRequest):
    """SSE 流式对话：执行过程中实时推送阶段进度（progress 事件），
    结束时推送完整结果（result 事件）。前端据此展示动态分析过程。"""
    import json

    thread_id = (
        req.thread_id or f"chat_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"
    )
    queue: asyncio.Queue = asyncio.Queue()

    async def progress_cb(event: dict):
        await queue.put({"event": "progress", "data": json.dumps(event, ensure_ascii=False)})

    async def run_graph():
        try:
            async with _graph_lock:
                current_graph = fin_graph
            result = await asyncio.wait_for(
                current_graph.run(
                    req.user_input,
                    thread_id,
                    user_id=req.user_id,
                    agents=req.agents,
                    progress_cb=progress_cb,
                ),
                timeout=300,
            )
            if result.get("status") == "error":
                await queue.put(
                    {
                        "event": "error",
                        "data": json.dumps(
                            {"error": result.get("result", "分析失败")},
                            ensure_ascii=False,
                        ),
                    }
                )
            else:
                await queue.put(
                    {
                        "event": "result",
                        "data": json.dumps(
                            _build_chat_response(result, thread_id),
                            ensure_ascii=False,
                            default=str,
                        ),
                    }
                )
        except asyncio.TimeoutError:
            await queue.put(
                {
                    "event": "error",
                    "data": json.dumps(
                        {"error": "分析超时了（5分钟），请换个问题或稍后再试"},
                        ensure_ascii=False,
                    ),
                }
            )
        except Exception as e:
            import traceback

            traceback.print_exc()
            await queue.put(
                {
                    "event": "error",
                    "data": json.dumps({"error": f"分析出错: {e}"}, ensure_ascii=False),
                }
            )
        finally:
            await queue.put(None)  # 结束哨兵

    async def event_generator():
        task = asyncio.create_task(run_graph())
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield item
        finally:
            if not task.done():
                task.cancel()

    return EventSourceResponse(
        event_generator(),
        ping=15,
        ping_message_factory=lambda: {"event": "ping", "data": "{}"},
    )


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
        test_llm = create_chat_model(
            api_key=req.api_key,
            base_url=req.base_url,
            model_name=req.model_name,
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


# ========== 行情缓存（stale-while-revalidate） ==========
# 行情数据源（AKShare 全量 A 股快照 / Tiingo 逐只拉取）单次都要几秒到几十秒。
# 策略：有缓存就立即返回（过期则返回旧数据并后台刷新）；只有冷启动才等待。
_MARKET_TTL_S = {"zh": 60, "us": 120}
_MARKET_CACHE: dict = {}          # key -> {"data": payload, "ts": float}
_MARKET_LOCKS: dict = {}          # key -> asyncio.Lock，防止并发重复拉取
_MARKET_REFRESHING: set = set()   # 正在后台刷新的 key
_BG_TASKS: set = set()            # 后台任务引用，防止 fire-and-forget 被 GC（review P3）


async def _fetch_market(market: str, limit: int) -> dict:
    """真正去数据源拉取（带超时，避免外部接口挂起拖死请求）。"""
    if market == "zh":
        return await asyncio.wait_for(_get_zh_market(limit), timeout=60)
    return await asyncio.wait_for(_get_us_market(), timeout=60)


_MARKET_SNAPSHOT_DIR = Path(__file__).resolve().parent / ".cache" / "market"


def _snapshot_path(key: str) -> Path:
    return _MARKET_SNAPSHOT_DIR / (key.replace(":", "_") + ".json")


def _save_market_snapshot(key: str, data: dict, ts: float) -> None:
    """把最近一次成功的行情落盘：后端重启后可立即展示，不必等数据源。"""
    try:
        _MARKET_SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        _snapshot_path(key).write_text(
            json.dumps({"ts": ts, "data": data}, ensure_ascii=False), encoding="utf-8"
        )
    except Exception as e:
        print(f"[market] 快照写入失败（不影响服务）: {e}")


def _load_market_snapshots() -> None:
    for key in ("zh:50", "us:50"):
        path = _snapshot_path(key)
        if not path.is_file():
            continue
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
            stocks = obj.get("data", {}).get("stocks") or []
            # 美股必须是完整一组（残缺快照不能当作有效数据展示）
            if key.startswith("us:") and len(stocks) < len(US_HOT_STOCKS):
                continue
            if stocks:
                _MARKET_CACHE[key] = {"data": obj["data"], "ts": float(obj["ts"])}
        except Exception:
            pass


async def _refresh_market(key: str, market: str, limit: int) -> dict:
    lock = _MARKET_LOCKS.setdefault(key, asyncio.Lock())
    async with lock:
        try:
            data = await _fetch_market(market, limit)
        except Exception as e:  # 超时、子进程异常或数据源异常
            data = {"success": False, "error": f"行情获取失败: {e}"}
        # 空结果（如数据源限流返回 429 导致 0 条）视为失败，不能当成功缓存
        if data.get("success") and not data.get("stocks"):
            data = {"success": False, "error": "行情数据源暂时没有返回数据（可能是接口限流），请稍后重试"}
        if data.get("success"):
            ts = time.time()
            _MARKET_CACHE[key] = {"data": data, "ts": ts}
            _save_market_snapshot(key, data, ts)
        return data


def _schedule_market_refresh(key: str, market: str, limit: int) -> None:
    if key in _MARKET_REFRESHING:
        return
    _MARKET_REFRESHING.add(key)

    async def _run():
        try:
            await _refresh_market(key, market, limit)
        finally:
            _MARKET_REFRESHING.discard(key)

    # 保留引用：fire-and-forget 任务无人引用时可能被 GC 中途取消（review P3）
    _t = asyncio.create_task(_run())
    _BG_TASKS.add(_t)
    _t.add_done_callback(_BG_TASKS.discard)


async def _prewarm_market_cache() -> None:
    """启动时先读落盘快照（立即可用），再后台预热行情缓存。"""
    _load_market_snapshots()

    async def _warm():
        for m in ("zh", "us"):
            try:
                await _refresh_market(f"{m}:50", m, 50)
            except Exception:
                pass

    _t = asyncio.create_task(_warm())
    _BG_TASKS.add(_t)
    _t.add_done_callback(_BG_TASKS.discard)


@app.get("/api/v1/market")
async def get_market(market: str = "zh", limit: int = 50):
    """
    获取实时行情数据（带缓存）
    - market=zh: A股实时行情（按涨跌幅绝对值排序）
    - market=us: 美股热门股票行情
    返回字段 cached / age_s / stale 说明数据新旧，前端据此提示"数据更新于 …"。
    """
    if market not in ("zh", "us"):
        return {"success": False, "error": "不支持的市场类型，请使用 zh 或 us"}

    key = f"{market}:{limit}"
    ttl = _MARKET_TTL_S[market]
    entry = _MARKET_CACHE.get(key)
    if entry and not entry["data"].get("stocks"):
        entry = None
    if entry:
        age = time.time() - entry["ts"]
        if age >= ttl:
            _schedule_market_refresh(key, market, limit)  # 旧数据先返回，后台刷新
        return {**entry["data"], "cached": True, "age_s": round(age, 1), "stale": age >= ttl}

    # 冷启动：只能等待首次拉取
    data = await _refresh_market(key, market, limit)
    if not data.get("success"):
        return data
    return {**data, "cached": False, "age_s": 0.0, "stale": False}


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


async def _get_us_market_tiingo():
    """美股兜底数据源：Tiingo（并发拉取，消耗免费额度）。Yahoo 失败/缺票时启用。"""
    try:
        import httpx
        from datetime import datetime, timedelta

        token = settings.tiingo_api_key
        if not token:
            return {"success": False, "error": "TIINGO_API_KEY 未配置"}

        # 获取最近 7 天的数据用于计算涨跌幅
        start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

        async def _one(client, symbol):
            try:
                url = (
                    f"https://api.tiingo.com/tiingo/daily/{symbol}/prices"
                    f"?startDate={start_date}&token={token}"
                )
                r = await client.get(url)
                data = r.json()
                if not isinstance(data, list) or len(data) < 2:
                    return None
                latest, prev = data[-1], data[-2]
                price = float(latest.get("close", 0))
                prev_close = float(prev.get("close", 0))
                change = price - prev_close
                change_percent = (change / prev_close * 100) if prev_close else 0
                volume = int(latest.get("volume", 0)) if latest.get("volume") else 0
                return {
                    "symbol": symbol,
                    "name": symbol,  # Tiingo 不返回名称，用 symbol 代替
                    "price": round(price, 2),
                    "change": round(change, 2),
                    "changePercent": round(change_percent, 2),
                    "volume": volume,
                    "dayHigh": float(latest.get("high", 0)),
                    "dayLow": float(latest.get("low", 0)),
                }
            except Exception:
                return None

        # 并发请求（原来是串行 await，每只股票一次往返）
        async with httpx.AsyncClient(timeout=15.0) as client:
            results = await asyncio.gather(*(_one(client, sym) for sym in US_HOT_STOCKS))
        stocks = [r for r in results if r]

        # 按涨跌幅绝对值排序
        stocks.sort(key=lambda x: abs(x["changePercent"]), reverse=True)

        return {"success": True, "market": "us", "source": "tiingo", "count": len(stocks), "stocks": stocks}
    except Exception as e:
        return {"success": False, "error": f"获取美股行情失败: {str(e)}"}


async def _get_us_market_yahoo():
    """美股主数据源：Yahoo Finance 直连（无需 key、无小时配额，见 Agent.md §7.14 额度分析）。"""
    from Data.providers.yahoo_us import fetch_us_quotes

    try:
        stocks = await fetch_us_quotes(US_HOT_STOCKS)
        if not stocks:
            return {"success": False, "error": "Yahoo 无返回"}
        stocks.sort(key=lambda x: abs(x["changePercent"]), reverse=True)
        return {"success": True, "market": "us", "source": "yahoo", "count": len(stocks), "stocks": stocks}
    except Exception as e:
        return {"success": False, "error": str(e)}


async def _get_us_market():
    """美股行情：Yahoo 直连优先，失败/缺票时兜底 Tiingo（不混用两个源）。"""
    try:
        primary = await _get_us_market_yahoo()
    except Exception as e:  # 主源异常也必须进入兜底，不能穿透成 500
        primary = {"success": False, "error": f"异常: {e}"}
    # 只接受完整结果：主源部分缺票时整体切到兜底源
    if primary.get("success") and len(primary.get("stocks") or []) >= len(US_HOT_STOCKS):
        return primary

    primary_err = primary.get("error") or f"仅返回 {len(primary.get('stocks') or [])}/{len(US_HOT_STOCKS)} 只"
    try:
        fallback = await _get_us_market_tiingo()
    except Exception as e:  # 兜底源异常同样不能穿透成 500
        fallback = {"success": False, "error": f"异常: {e}"}
    if fallback.get("success") and len(fallback.get("stocks") or []) >= len(US_HOT_STOCKS):
        return fallback

    fallback_err = fallback.get("error") or f"仅返回 {len(fallback.get('stocks') or [])}/{len(US_HOT_STOCKS)} 只"
    return {"success": False, "error": f"美股行情暂不可用（Yahoo：{primary_err}；Tiingo：{fallback_err}）"}


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
    """同步回测接口（保留，但【不作为演示路径】）。

    慢：逐日同步调用 LLM，6 个交易日就可能超过 290 秒，HTTP 请求会阻塞。
    前端只使用 /api/v1/backtest-stream（流式进度 + 可取消）。
    """
    try:
        import asyncio

        os.environ["FINGENT_BACKTEST_STRICT"] = "1"

        mc = get_model_config()
        temp_model = create_chat_model(
            api_key=mc.api_key,
            base_url=mc.base_url,
            model_name=mc.model_name,
            temperature=req.temperature,
        )
        temp_checkpointer, temp_cleanup = await get_checkpointer_async()
        temp_graph = assemble_fin_graph(
            temp_model, temp_checkpointer, agent_names=BACKTEST_AGENT_NAMES
        )

        try:
            # GraphSignalStrategy 内部用 run_until_complete 驱动图，必须在「没有运行中事件循环」的线程里同步执行。
            # 注意：run_backtest 是同步函数，不能包在 async 函数（asyncio.run）里调用，否则循环冲突。
            # 做法：工作线程里先 asyncio.run 只加载数据，再同步调用 run_backtest。
            from Trade.runner import load_price_dataframe, run_backtest

            def _backtest_sync():
                data = asyncio.run(
                    load_price_dataframe(
                        symbol=req.symbol,
                        start=req.start,
                        end=req.end,
                        resample_freq=req.interval,
                    )
                )
                return run_backtest(
                    fin_graph=temp_graph,
                    data=data,
                    symbol=req.symbol,
                    initial_cash=req.initial_cash,
                    commission=req.commission,
                    slippage_perc=req.slippage,
                    min_confidence=req.min_confidence,
                    rebalance_threshold=req.rebalance_threshold,
                    printlog=(not req.quiet),
                    audit_path=req.audit_path,
                )

            result = await asyncio.to_thread(_backtest_sync)
        finally:
            if temp_cleanup:
                await temp_cleanup()

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
        temp_cleanup = None
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
            temp_model = create_chat_model(
                api_key=mc.api_key,
                base_url=mc.base_url,
                model_name=mc.model_name,
                temperature=req.temperature,
            )
            temp_checkpointer, temp_cleanup = get_checkpointer()
            temp_graph = assemble_fin_graph(
                temp_model, temp_checkpointer, agent_names=BACKTEST_AGENT_NAMES
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
            try:
                if temp_cleanup:
                    temp_cleanup()
            except Exception:
                pass

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


# ========== Agent 角色管理 API ==========
# 底层角色的身份/协议/输出格式都在 LLM/skills/<name>/SKILL.md，
# 这里提供只读/编辑接口，保存后热重建 FinGraph 立即生效。

from fastapi import HTTPException


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """全局兜底：任何未捕获异常都返回结构化 JSON（HTTP 200 + success=False），
    前端按 success 字段展示错误气泡，不会出现裸 500/白屏。"""
    import traceback

    traceback.print_exc()
    return JSONResponse(
        status_code=200,
        content={"success": False, "error": f"服务器内部错误: {exc}"},
    )


import yaml
from LLM import registry as _agent_registry

_SKILL_NAME_RE = __import__("re").compile(r"^[a-z0-9][a-z0-9-_]*$")


def _skill_file_for(name: str):
    """返回 SKILL.md 路径；agent 无技能文件时返回 None。校验名字防目录穿越。"""
    if not _SKILL_NAME_RE.match(name or ""):
        raise HTTPException(status_code=400, detail=f"非法 agent 名: {name}")
    path = _agent_registry.SKILLS_DIR / name / "SKILL.md"
    return path if path.is_file() else None


def _parse_frontmatter(text: str) -> dict:
    try:
        parts = text.split("---", 2)
        if len(parts) < 3:
            return {}
        return yaml.safe_load(parts[1]) or {}
    except yaml.YAMLError:
        return {}


@app.get("/api/v1/agents")
async def list_agents():
    """列出全部已注册 agent（名字/描述/工具/技能文件状态）"""
    agents = []
    for name, info in _agent_registry.all_agents().items():
        skill_path = _agent_registry.SKILLS_DIR / (info.skill_name or name) / "SKILL.md"
        agents.append(
            {
                "name": name,
                "description": info.description,
                "allowed_tools": list(info.allowed_tools),
                "builtin": (info.skill_name or "") in (
                    "technical-nerd",
                    "morefit",
                    "risk-manager",
                    "sentiment-analyzer",
                ),
                "editable": skill_path.is_file(),
            }
        )
    return {"success": True, "agents": agents}


@app.get("/api/v1/agents/{name}")
async def get_agent_skill(name: str):
    """读取某个 agent 的 SKILL.md 全文"""
    path = _skill_file_for(name)
    if path is None:
        raise HTTPException(status_code=404, detail=f"agent {name} 没有可编辑的技能文件")
    content = path.read_text(encoding="utf-8")
    return {
        "success": True,
        "name": name,
        "content": content,
        "frontmatter": _parse_frontmatter(content),
    }


class SkillUpdateRequest(BaseModel):
    content: str


@app.put("/api/v1/agents/{name}")
async def update_agent_skill(name: str, req: SkillUpdateRequest):
    """保存 SKILL.md → 备份原文件 → 刷新注册表 → 热重建 FinGraph（立即生效）"""
    global fin_graph

    path = _skill_file_for(name)
    if path is None:
        raise HTTPException(status_code=404, detail=f"agent {name} 没有可编辑的技能文件")

    if not req.content or len(req.content) > 100_000:
        raise HTTPException(status_code=400, detail="内容为空或超过 100KB")

    # frontmatter 合法性校验（name 必须和目录一致，避免保存后注册表错乱）
    front = _parse_frontmatter(req.content)
    if front:
        fm_name = front.get("name")
        if fm_name and fm_name != name:
            raise HTTPException(
                status_code=400,
                detail=f"frontmatter name '{fm_name}' 与 agent '{name}' 不一致",
            )
    elif not req.content.lstrip().startswith("---"):
        raise HTTPException(status_code=400, detail="SKILL.md 需要以 --- frontmatter --- 开头")

    # 备份原文件（演示/回滚保险）
    backup = path.with_suffix(".md.bak")
    try:
        backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    except Exception:
        pass

    path.write_text(req.content, encoding="utf-8")

    # 刷新注册表描述 + 热重建 graph
    try:
        _agent_registry.discover_skills()
        async with _graph_lock:
            fin_graph = build_fin_graph(get_model_config())
    except Exception as e:
        return {"success": False, "error": f"已保存文件，但热重建失败（重启服务后生效）: {e}"}

    return {
        "success": True,
        "message": f"已保存 {name}/SKILL.md 并热重建，新的对话立即生效",
        "backup": str(backup),
    }


@app.post("/api/v1/agents/reload")
async def reload_agents():
    """重新扫描技能目录并热重建 FinGraph（手动改文件后用）"""
    global fin_graph

    try:
        discovered = _agent_registry.discover_skills()
        async with _graph_lock:
            fin_graph = build_fin_graph(get_model_config())
        return {
            "success": True,
            "message": f"已重载 {len(discovered)} 个 agent",
            "agents": list(discovered.keys()),
        }
    except Exception as e:
        return {"success": False, "error": f"重载失败: {e}"}


# ========== 可靠性评测 API（只读） ==========
# 数据来源：Eval/runs/<run_id>/{report.json,trace.jsonl}（由 Eval/reliability.py 生成）。
# 只读、防路径穿越：run_id 必须匹配 ^\d{8}_\d{6}$，且解析后的目录必须在 Eval/runs 下并真实存在。

_EVAL_RUN_ID_RE = __import__("re").compile(r"^\d{8}_\d{6}$")
_EVAL_RUNS_DIR = Path(__file__).resolve().parent / "Eval" / "runs"


def _resolve_eval_run(run_id: str) -> Path:
    """校验 run_id 并返回其目录；非法 → 400，不存在 → 404。"""
    if not _EVAL_RUN_ID_RE.match(run_id or ""):
        raise HTTPException(status_code=400, detail=f"非法 run_id: {run_id}")
    runs_root = _EVAL_RUNS_DIR.resolve()
    run_dir = (_EVAL_RUNS_DIR / run_id).resolve()
    if run_dir == runs_root or runs_root not in run_dir.parents or not run_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"评测运行不存在: {run_id}")
    return run_dir


@app.get("/api/v1/eval/runs")
async def list_eval_runs():
    """扫描 Eval/runs/*/report.json，按 run_id 倒序返回摘要列表。"""
    runs = []
    if _EVAL_RUNS_DIR.is_dir():
        for d in _EVAL_RUNS_DIR.iterdir():
            if not (d.is_dir() and _EVAL_RUN_ID_RE.match(d.name)):
                continue
            report_path = d / "report.json"
            if not report_path.is_file():
                continue
            try:
                rep = json.loads(report_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            cases = rep.get("cases", []) or []
            runs.append(
                {
                    "run_id": d.name,
                    "generated_at": rep.get("generated_at"),
                    "k": rep.get("k"),
                    "total_runs": rep.get("total_runs", sum(c.get("runs", 0) for c in cases)),
                    "case_count": len(cases),
                    "pipeline_ok": sum(c.get("pipeline_ok", 0) for c in cases),
                    "strict_ok": sum(c.get("strict_ok", 0) for c in cases),
                    "cases": [c.get("case_id") for c in cases],
                }
            )
    runs.sort(key=lambda r: r["run_id"], reverse=True)
    return {"success": True, "count": len(runs), "runs": runs}


@app.get("/api/v1/eval/runs/{run_id}")
async def get_eval_run(run_id: str):
    """返回某次评测的 report.json 全文。"""
    report_path = _resolve_eval_run(run_id) / "report.json"
    if not report_path.is_file():
        raise HTTPException(status_code=404, detail=f"缺少 report.json: {run_id}")
    return {
        "success": True,
        "run_id": run_id,
        "report": json.loads(report_path.read_text(encoding="utf-8")),
    }


@app.get("/api/v1/eval/runs/{run_id}/trace")
async def get_eval_trace(run_id: str):
    """返回结构化运行分组（含逐步事件时间线），复用 Eval.reliability.parse_trace。"""
    trace_path = _resolve_eval_run(run_id) / "trace.jsonl"
    if not trace_path.is_file():
        raise HTTPException(status_code=404, detail=f"缺少 trace.jsonl: {run_id}")
    from Eval.reliability import parse_trace

    return {"success": True, "run_id": run_id, "runs": parse_trace(trace_path)}


@app.get("/api/v1/eval/runs/{run_id}/log")
async def get_eval_log(run_id: str, level: Optional[str] = None):
    """返回结构化运行日志 run_log.jsonl（JSON Lines 逐行解析）。可按 level 过滤。
    坏行不会让接口失败，只计入 skipped_lines。"""
    log_path = _resolve_eval_run(run_id) / "run_log.jsonl"
    if not log_path.is_file():
        # 早期批次没有结构化日志：这是正常情况，返回 200 + available=false，不当作错误
        return {"success": True, "run_id": run_id, "available": False, "count": 0, "skipped_lines": 0, "rows": []}
    rows, skipped = [], 0
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            skipped += 1
            continue
        if level and row.get("level") != level.upper():
            continue
        rows.append(row)
    return {"success": True, "run_id": run_id, "available": True, "count": len(rows), "skipped_lines": skipped, "rows": rows}


@app.get("/api/v1/eval/runs/{run_id}/report-md")
async def get_eval_report_md(run_id: str):
    """返回 report.md 原文（前端渲染为 Markdown）。"""
    report_md = _resolve_eval_run(run_id) / "report.md"
    if not report_md.is_file():
        return {"success": True, "run_id": run_id, "available": False, "markdown": ""}
    return {"success": True, "run_id": run_id, "available": True, "markdown": report_md.read_text(encoding="utf-8")}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=settings.api_host, port=settings.api_port)
