from typing import TypedDict, Any, Optional, List, Dict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

try:
    from config import get_settings as _get_settings
    _config = _get_settings()
except Exception:
    _config = None

try:
    from LLM.preprocess import Preprocessor
    from LLM.router import Router
    from LLM.agent import TECHNICAL_NERD, Morefit
    from LLM.unified_stock_tools import (
        get_stock_price,
        get_stock_basic_info,
        get_stock_company_info,
        get_stock_financial_report_links,
    )
except ImportError:
    from preprocess import Preprocessor
    from router import Router
    from agent import TECHNICAL_NERD, Morefit
    from unified_stock_tools import (
        get_stock_price,
        get_stock_basic_info,
        get_stock_company_info,
        get_stock_financial_report_links,
    )

try:
    from Data.memory import get_memory_manager
except ImportError:
    get_memory_manager = None
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
import json
import asyncio
import re
import contextvars

# ── 执行进度推送（SSE 用）：contextvar 保证并发请求之间互不串扰 ──
_progress_cb_var: contextvars.ContextVar = contextvars.ContextVar(
    "fingent_progress_cb", default=None
)


def emit_progress(event: dict) -> None:
    """在任意节点（同步/异步）中推送一条进度事件；无回调时零开销。

    通过 create_task 派发，同步节点（route/clarify）里也能安全调用。
    """
    cb = _progress_cb_var.get()
    if cb is None:
        return

    async def _fire():
        try:
            r = cb(event)
            if asyncio.iscoroutine(r):
                await r
        except Exception:
            pass  # 进度推送失败绝不影响主流程

    try:
        asyncio.get_running_loop().create_task(_fire())
    except RuntimeError:
        pass  # 无事件循环（CLI 同步场景），跳过进度

load_dotenv()
qianwen_api_key = os.getenv("QIANWEN_API_KEY")


# =============================================================================
# Agent 注册表 — agent 唯一来源（LLM/registry.py），不再硬编码 AGENT_CONFIG
# =============================================================================
from LLM.registry import all_agents as _registry_all_agents


def _agent_config(name: str) -> dict:
    """注册表里的 agent 配置；未知 agent 返回默认 json 解析。"""
    info = _registry_all_agents().get(name)
    return {
        "output_parser": info.output_parser if info else "json",
        "description": info.description if info else name,
    }


# =============================================================================
# State 定义 — 用 agent_results 字典取代硬编码的 vote 字段
# =============================================================================
class GraphStatus(TypedDict):
    input: str
    parse_result: Optional[dict]
    result: Any
    status: str
    thread_id: str
    next_agent: Optional[str]
    current_agent: Optional[str]  # 当前执行的 Agent 名称
    selected_agents: Optional[list]  # 本次分析参与的 agent 名单（显式>语义>兜底）
    route_info: Optional[dict]  # 路由解释 {mode, agents, reason, source}，供前端思维链
    clarification_count: int
    agent_results: Optional[Dict[str, dict]]  # 各 Agent 的完整结果 {agent_name: result}
    final_decision: Optional[dict]
    all_tool_chains: Optional[list]
    detailed_analysis: Optional[dict]
    tool_chain: Optional[list]
    agent_name: Optional[str]
    stock: Optional[str]


# =============================================================================
# FinGraph — 配置驱动的 LangGraph
# =============================================================================
class FinGraph:
    def __init__(
        self,
        preprocessor: Preprocessor,
        router: Router,
        agent: dict = None,
        checkpointer=None,
        memory_manager=None,
    ):
        self.preprocessor = preprocessor
        self.router = router
        self.agent = agent or {}
        self.checkpointer = checkpointer or MemorySaver()
        self.memory_manager = memory_manager
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(GraphStatus)

        # 固定节点
        workflow.add_node("preprocess", self._preprocess)
        workflow.add_node("route", self._route)
        workflow.add_node("execute_agent", self._execute_agent)  # 统一单 Agent 执行
        workflow.add_node("clarify_node", self._clarify)
        workflow.add_node("ALL", self._run_all_agents)  # 并行多 Agent
        workflow.add_node("voting", self._voting)  # 通用投票汇总
        workflow.add_node("chat_node", self._chat)  # 闲聊/通用问题直接回复（不走分析流程）
        workflow.add_node("sector_node", self._sector)  # 板块/市场问题（未指定个股）直接回答，不走澄清

        workflow.set_entry_point("preprocess")
        workflow.add_edge("preprocess", "route")

        # 动态路由映射：固定节点 + 注册表名兼容（路由结果以 execute_agent/ALL/clarify 为准）
        route_mapping = {name: "execute_agent" for name in _registry_all_agents()}
        route_mapping["execute_agent"] = "execute_agent"
        route_mapping["ALL"] = "ALL"
        route_mapping["clarify_node"] = "clarify_node"
        route_mapping["chat_node"] = "chat_node"
        route_mapping["sector_node"] = "sector_node"
        route_mapping["end"] = END

        workflow.add_conditional_edges(
            "route",
            lambda state: state.get("next_agent", "end"),
            route_mapping,
        )

        workflow.add_edge("execute_agent", END)
        workflow.add_edge("ALL", "voting")
        workflow.add_edge("voting", END)
        workflow.add_edge("clarify_node", END)
        workflow.add_edge("chat_node", END)
        workflow.add_edge("sector_node", END)

        return workflow.compile(checkpointer=self.checkpointer)

    # -------------------------------------------------------------------------
    # 闲聊节点 — 问候/通用问题直接回复，不进入分析流程
    # -------------------------------------------------------------------------
    _CASUAL_CHAT_RE = re.compile(
        r"^\s*(你好|您好|嗨|哈喽|hallo|hello|hi|hey|早|早上好|上午好|中午好|下午好|晚上好|晚安|"
        r"谢谢|感谢|多谢|thx|thanks|thank you|辛苦|再见|拜拜|bye|"
        r"你是谁|你能做什么|你会什么|你能干嘛|你有什么用|自我介绍|介绍.*自己|help|帮助)\s*[！!。.~，,]*\s*$",
        re.IGNORECASE,
    )

    def _is_casual_chat(self, text: str) -> bool:
        """纯闲聊/问候/求助（全串匹配，避免误判带股票名的句子）"""
        return bool(self._CASUAL_CHAT_RE.match((text or "").strip()))

    async def _chat(self, state: GraphStatus) -> dict:
        """直接对话：用底层模型自然回复，不走 agent / 投票流程"""
        print(f"\n┌─ [chat_node] 闲聊/通用问题直接回复")
        emit_progress({"stage": "chat"})
        user_input = state["input"]

        model = getattr(self.preprocessor, "model", None)
        if model is None:
            return {
                "result": "你好！我是 FinGent，可以帮你分析 A 股和美股。直接告诉我股票名称或代码就可以开始，比如「英伟达值得买吗」。",
                "status": "completed",
            }

        system_prompt = (
            "你是 FinGent，一个 AI 股票分析助手。用简洁、自然、口语化的中文回答。"
            "正常陪用户聊天；介绍自己时可以提到你能做：个股技术面/基本面/风险/舆情分析、"
            "多 Agent 投票决策、策略回测。如果用户问到具体股票，引导他直接说股票名称或代码。"
            "回答控制在几句话以内。"
        )
        try:
            resp = await model.ainvoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_input),
                ]
            )
            answer = resp.content if isinstance(resp.content, str) else str(resp.content)
        except Exception as e:
            print(f"│  ⚠️ 闲聊回复失败: {e}")
            answer = "你好！我是 FinGent，可以帮你分析 A 股和美股。直接告诉我股票名称或代码就可以开始～"

        print(f"└─ 直接回复完成")
        return {"result": answer, "status": "completed"}

    _SECTOR_TECH_RE = re.compile(
        r"科技|纳指|纳斯达克|美股|芯片|半导体|算力|人工智能|\bAI\b|大厂|英伟达|nvda", re.IGNORECASE
    )

    async def _sector(self, state: GraphStatus) -> dict:
        """板块/市场层面问题（未指定个股）：基于实时背景给框架性回答，不强制解析个股。"""
        print(f"\n┌─ [sector_node] 板块/市场问答（未指定个股）")
        emit_progress({"stage": "sector"})
        user_input = state["input"]

        from Data.market_context import tech_market_context

        context_text, stats = "", None
        if self._SECTOR_TECH_RE.search(user_input or ""):
            context_text, stats = await tech_market_context()
        print(f"│  实时背景: {'有' if context_text else '无（数据源不可用，不编造数字）'}")

        model = getattr(self.preprocessor, "model", None)
        if model is None:
            return {"result": self._sector_fallback_text(), "status": "completed"}

        context_block = (
            f"【实时背景（仅此处数字可引用）】\n{context_text}\n"
            if context_text else
            "【实时背景】暂无可用数据。不要编造任何价格、涨跌或估值数字。\n"
        )
        system_prompt = (
            "你是 FinGent，一个 AI 股票分析助手。用户在问板块或市场层面的问题，没有指定具体股票。"
            "请用简洁、口语化的中文回答，不超过 8 行：\n"
            "1) 先给出判断框架：估值水平、业绩兑现、资金与情绪、利率与监管、仓位与风险承受能力；\n"
            "2) 结合下方实时背景说明当前可观察到的信号（只能引用背景中的数字）；单日涨跌只代表当天表现，不能据此断言资金流向或趋势；\n"
            "3) 明确说明'还能不能买'没有统一答案，取决于个人周期和风险承受能力，不构成投资建议；\n"
            "4) 最后一句引导：想看具体标的，直接说股票名称或代码（如「英伟达」「AAPL」），我会做多 Agent 完整分析。\n\n"
            + context_block
        )
        try:
            resp = await model.ainvoke(
                [SystemMessage(content=system_prompt), HumanMessage(content=user_input)]
            )
            answer = resp.content if isinstance(resp.content, str) else str(resp.content)
        except Exception as e:
            print(f"│  ⚠️ 板块回答失败: {e}")
            answer = self._sector_fallback_text(context_text)

        print(f"└─ 板块回答完成")
        return {"result": answer, "status": "completed"}

    @staticmethod
    def _sector_fallback_text(context_text: str = "") -> str:
        head = "这个问题没有统一答案，要看估值、业绩兑现、资金情绪和你自己的周期与风险承受能力。"
        tail = "想看具体标的，直接告诉我股票名称或代码（如「英伟达」「AAPL」），我会做完整分析。"
        if context_text:
            return f"{head}\n当前参考：\n{context_text}\n{tail}"
        return f"{head}\n{tail}"

    # -------------------------------------------------------------------------
    # 预处理节点
    # -------------------------------------------------------------------------
    async def _preprocess(self, state: GraphStatus) -> dict:
        print(f"\n┌─ [preprocess] 预处理节点")
        emit_progress({"stage": "preprocess"})
        print(f"│  输入: '{state['input']}'")
        try:
            result = await self.preprocessor.async_parse(
                state["input"], thread_id=state["thread_id"]
            )
        except Exception as e:
            # 兜底：LLM 解析失败（网络/JSON 解析等）时走本地规则，保证请求不崩
            print(f"│  ⚠️ LLM 解析失败，启用本地兜底解析: {str(e)[:80]}")
            result = self._fallback_parse(state["input"])
        print(
            f"│  解析完成: intent={result.get('intent', {}).get('type')}, stock={result.get('stock')}"
        )
        print(f"└─ 输出: status={result['status']}")
        return {"parse_result": result, "status": result["status"]}

    _FALLBACK_TICKER_RE = re.compile(r"\b([A-Za-z]{2,5})\b")

    def _fallback_parse(self, user_input: str) -> dict:
        """LLM 解析失败时的本地兜底：尽力提取股票代码，提不出则进入澄清"""
        text = (user_input or "").strip()
        # 回测格式输入仍然可以精确解析
        bt = getattr(self.preprocessor, "_parse_backtest_format", None)
        if bt:
            try:
                parsed = bt(text)
                if parsed:
                    return parsed
            except Exception:
                pass

        code = None
        m = re.search(r"\b(\d{6})\b", text)
        if m:
            code = m.group(1)
        else:
            m = self._FALLBACK_TICKER_RE.search(text)
            if m:
                candidate = m.group(1).upper()
                if candidate not in {"AI", "OK", "API", "HTTP", "USD", "HKD", "CNY"}:
                    code = candidate

        if code:
            return {
                "status": "ready",
                "intent": {"type": "MARKET_DATA"},
                "entities": {"symbols": [code], "names": [], "code": [code]},
                "agent_selection": [],
                "time_range": {"start": "", "end": ""},
                "original_input": user_input,
                "clarification": None,
            }
        # 提不出个股：不是死胡同，交给板块/通用回答（会引导用户给出股票名称或代码）
        return {
            "status": "ready",
            "intent": {"type": "SUGGESTIONS"},
            "entities": {"symbols": [], "names": [], "code": []},
            "agent_selection": [],
            "time_range": {"start": "", "end": ""},
            "original_input": user_input,
            "clarification": None,
        }

    # -------------------------------------------------------------------------
    # 路由节点
    # -------------------------------------------------------------------------
    def _route(self, state: GraphStatus) -> dict:
        print(f"\n┌─ [route] 路由节点")
        parse_result = state["parse_result"] or {}
        print(f"│  parse_result: {parse_result}")

        intent_type = (parse_result.get("intent") or {}).get("type", "")

        # 0. 闲聊/通用问题 → 直接对话（最优先，避免普通聊天被强行拉去分析/澄清）
        if intent_type == "CHAT" or self._is_casual_chat(state["input"]):
            print(f"└─ 路由到: -> chat_node (直接回复)")
            info = self._make_route_info(
                "chat", [], "闲聊/通用问题，直接对话，不进入分析流程", "chat"
            )
            return {"next_agent": "chat_node", "status": "routed", "route_info": info}

        # 1. 需要澄清
        if parse_result.get("status") == "clarification_needed":
            print(f"└─ 路由到: -> clarify_node")
            clarify = parse_result.get("clarification") or {}
            reason = clarify.get("message") or "信息不足，需要用户澄清"
            info = self._make_route_info("clarify", [], reason, "clarify")
            return {
                "next_agent": "clarify_node",
                "status": "need_clarification",
                "route_info": info,
            }

        # 2. 缺少实体
        entities = parse_result.get("entities", {})
        codes = entities.get("code", [])
        symbols = entities.get("symbols", [])
        names = entities.get("names", [])

        if not codes and not symbols and not names:
            # 没有个股实体不是死胡同：板块/市场问题直接给框架性回答（可带实时背景），
            # 个股分析需要时再引导用户给出名称或代码。
            print(f"│  未识别到公司实体 -> 板块/通用回答（不强制个股）")
            print(f"└─ 路由到: -> sector_node")
            info = self._make_route_info(
                "sector", [], "未指定具体股票，按板块/市场问题直接回答", "sector"
            )
            return {
                "next_agent": "sector_node",
                "status": "routed",
                "route_info": info,
            }

        # 3. 选择本次参与的 agent：显式（API/CLI）> 语义（preprocess）> 旧 Router 兜底
        stock = self._pick_stock(entities)
        selected, source = self._select_agents(state, parse_result)
        reason = self._route_reason(source, selected, parse_result, intent_type)
        print(f"│  公司: {stock}")
        print(f"│  本次参与 Agent: {selected if selected else '（无，走澄清）'}")
        print(f"│  路由依据[{source}]: {reason}")

        if len(selected) == 1:
            print(f"└─ 路由到: -> {selected[0]} (单 Agent)")
            info = self._make_route_info("single", selected, reason, source)
            return {
                "next_agent": "execute_agent",
                "current_agent": selected[0],
                "status": "routed",
                "stock": stock,
                "selected_agents": selected,
                "route_info": info,
            }
        elif len(selected) > 1:
            print(f"└─ 路由到: -> ALL ({len(selected)} 个 Agent 并行投票)")
            info = self._make_route_info("multi", selected, reason, source)
            return {
                "next_agent": "ALL",
                "current_agent": None,
                "status": "routed",
                "stock": stock,
                "selected_agents": selected,
                "route_info": info,
            }

        print("└─ 无可参与 Agent，进入澄清")
        info = self._make_route_info(
            "clarify", [], "问题已识别到股票，但没有匹配到可用的分析 Agent", "none"
        )
        return {
            "next_agent": "clarify_node",
            "status": "need_agent",
            "selected_agents": [],
            "route_info": info,
        }

    def _select_agents(
        self, state: GraphStatus, parse_result: Optional[dict]
    ) -> tuple[list, str]:
        """确定本次运行的 agent 名单（显式 > 语义 > 旧 Router 静态表兜底）。

        显式为硬意图（如 API agents=['risk-manager']），语义来自 preprocess 的
        agent_selection（LLM 依据注册表 description 选择），两者都为空时退回
        Router.ROUTES 的旧静态映射。未装配/未注册的名字被过滤并告警。

        返回 ``(有效 agent 名单, 来源)``，来源为 explicit / semantic / fallback
        三选一，供路由解释（route_info）透出到前端思维链。
        """
        explicit = state.get("selected_agents") or []
        semantic = (parse_result or {}).get("agent_selection") or []

        if explicit:
            names, source = list(explicit), "explicit"
        elif semantic:
            names, source = list(semantic), "semantic"
        else:
            source = "fallback"
            routed = self.router.route(parse_result)
            if routed == "ALL":
                names = [n for n in _registry_all_agents() if n in self.agent]
            elif routed in self.agent:
                names = [routed]
            else:
                names = []

        valid = [n for n in names if n in self.agent]
        invalid = set(names) - set(valid)
        if invalid:
            print(f"│  ⚠️ 忽略未装配/未注册的 agent: {invalid}")
        return valid, source

    def _route_reason(
        self,
        source: str,
        selected: list,
        parse_result: Optional[dict],
        intent_type: str,
    ) -> str:
        """把路由来源翻译成一句人话，供前端思维链展示。"""
        joined = "、".join(selected) if selected else "（无）"
        if source == "semantic":
            raw = (parse_result or {}).get("agent_selection_reason")
            if isinstance(raw, str) and raw.strip():
                return raw.strip()
            return f"语义匹配：依据问题意图选择 {joined}"
        if source == "explicit":
            return f"用户显式指定：{joined}"
        return f"按意图 {intent_type or 'UNKNOWN'} 兜底路由到 {joined}"

    def _make_route_info(
        self, mode: str, agents: list, reason: str, source: str
    ) -> dict:
        """构造 route_info 并推送 route 进度事件（两者共用同一份内容）。"""
        info = {
            "mode": mode,
            "agents": list(agents or []),
            "reason": reason,
            "source": source,
        }
        emit_progress({"stage": "route", **info})
        return info

    # -------------------------------------------------------------------------
    # 统一单 Agent 执行节点（所有单 Agent 模式走这里）
    # -------------------------------------------------------------------------
    async def _execute_agent(self, state: GraphStatus) -> dict:
        agent_name = state.get("current_agent")
        print(f"\n┌─ [execute_agent] 统一执行节点 | Agent: {agent_name}")

        if not agent_name or agent_name not in _registry_all_agents():
            print(f"│  错误: Agent '{agent_name}' 未在注册表中定义")
            return {
                "result": f"未知 Agent: {agent_name}",
                "status": "error",
                "agent_name": agent_name,
                "tool_chain": [],
            }

        # 提取股票信息
        stock = self._extract_stock(state)
        print(f"│  分析股票: {stock}")
        print(f"│  用户问题: '{state['input']}'")
        print(f"│  调用 {agent_name} Agent 分析中...")

        # 统一底层执行
        result = await self._run_single_agent_async(
            agent_name,
            stock,
            state,
            user_id=state.get("user_id", "anonymous"),
        )

        # 生成文本结果（根据解析器类型）
        config = _agent_config(agent_name)
        if config.get("output_parser") == "json":
            result_text = (
                f"## 📊 {agent_name} 分析 - {result.get('symbol', stock)}\n\n"
                f"**投票**: {result.get('vote', 'HOLD')} | "
                f"**建议仓位**: {result.get('target_position_pct', 0.5)*100:.0f}% | "
                f"**置信度**: {result.get('confidence', 0.5)*100:.0f}%\n\n"
                f"**分析理由**:\n{result.get('reason', '暂无分析')}"
            )
        else:
            result_text = result.get("reason", "") or result.get("raw_output", "")

        print(f"│  ✓ Agent 分析完成")
        return {
            "result": result_text,
            "status": "completed",
            "agent_name": agent_name,
            "tool_chain": result.get("tool_chain", []),
            "stock": result.get("symbol", stock),
            "agent_results": {agent_name: result},
        }

    # -------------------------------------------------------------------------
    # 统一底层：执行单个 Agent（异步，被 _execute_agent 和 _run_all_agents 共用）
    # -------------------------------------------------------------------------
    async def _run_single_agent_async(
        self,
        agent_name: str,
        stock: str,
        state: GraphStatus,
        user_id: Optional[str] = None,
    ) -> dict:
        agent = self.agent.get(agent_name)
        config = _agent_config(agent_name)

        if not agent:
            return {
                "symbol": stock,
                "vote": "HOLD",
                "target_position_pct": 0.5,
                "confidence": 0.5,
                "reason": f"{agent_name}未初始化",
                "error": True,
                "tool_chain": [],
                "raw_output": "",
                "agent_name": agent_name,
            }

        # 构造提示词 — 优先用 Agent 自己的 format_prompt，否则走默认模板
        if hasattr(agent, "format_prompt"):
            prompt = agent.format_prompt(stock, state["input"])
        else:
            prompt = f"请分析股票：{stock}\n\n用户问题：{state['input']}"
        agent_thread_id = f"{state['thread_id']}_{agent_name}_{id(state)}"

        emit_progress({"stage": "agent_start", "agent": agent_name})

        # 执行 Agent（传入 stock 和 user_id 以触发 L3 记忆注入）
        # 单个 Agent 失败不得拖垮整个请求：异常时兜底为「观望」结果
        try:
            response = await agent.achat(
                prompt,
                thread_id=agent_thread_id,
                stock=stock,
                user_id=user_id,
            )
        except Exception as e:
            print(f"│  ❌ {agent_name} 分析失败，已兜底为观望: {str(e)[:100]}")
            emit_progress(
                {"stage": "agent_done", "agent": agent_name, "vote": "HOLD", "error": True}
            )
            return {
                "symbol": stock,
                "vote": "HOLD",
                "target_position_pct": 0.3,
                "confidence": 0.3,
                "reason": f"{agent_name} 分析暂时不可用（{str(e)[:80]}），暂持观望。",
                "error": True,
                "tool_chain": [],
                "raw_output": "",
                "agent_name": agent_name,
            }

        # 收集工具调用链
        tool_chain = self._extract_tool_chain(response.messages)
        print(
            f"│  📋 工具调用链 (共 {len(tool_chain)} 步, "
            f"{sum(1 for t in tool_chain if t['type'] == 'tool_call')} 个工具)"
        )

        # 解析输出
        output_parser = config.get("output_parser")
        parsed = self._parse_agent_output(response.final_answer, output_parser, stock)
        parsed["tool_chain"] = tool_chain
        parsed["raw_output"] = response.final_answer
        parsed["agent_name"] = agent_name

        emit_progress(
            {
                "stage": "agent_done",
                "agent": agent_name,
                "vote": parsed.get("vote", "HOLD"),
            }
        )

        return parsed

    # -------------------------------------------------------------------------
    # 辅助：从 State 提取股票信息
    # -------------------------------------------------------------------------
    def _extract_stock(self, state: GraphStatus) -> str:
        entities = state.get("parse_result", {}).get("entities", {})
        return self._pick_stock(entities or {})

    def _pick_stock(self, entities: dict) -> str:
        """从解析实体中选股。

        背景：preprocess(LLM) 会给美股公司输出其本地上市代码（如 SK海力士→000660/KOSPI），
        该代码在本项目（仅支持 US/CN）内无效。因此当 code 是 6 位数字、
        而名称能命中美股映射表（含中文别名）时，优先改用名称（后续 _resolve 会查表）。
        """
        try:
            from Data.providers import us_stock
        except ImportError:
            us_stock = None

        codes = entities.get("code", [])
        symbols = entities.get("symbols", [])
        names = entities.get("names", [])

        code = codes[0] if codes else None
        if (
            code
            and str(code).isdigit()
            and len(str(code)) == 6
            and names
            and us_stock is not None
        ):
            try:
                if us_stock.input2Symbol(names[0]):
                    return names[0]
            except Exception:
                pass

        return (
            code
            if code
            else symbols[0] if symbols else names[0] if names else "未知股票"
        )

    # -------------------------------------------------------------------------
    # 辅助：从消息列表提取工具调用链（统一逻辑，只写一次）
    # -------------------------------------------------------------------------
    def _extract_tool_chain(self, messages) -> list:
        tool_chain = []
        for msg in messages:
            msg_type = type(msg).__name__
            if msg_type == "HumanMessage":
                content = (
                    msg.content[:60] + "..." if len(msg.content) > 60 else msg.content
                )
                tool_chain.append({"type": "input", "content": content})
            elif msg_type == "AIMessage":
                tool_calls = getattr(msg, "tool_calls", None)
                if tool_calls:
                    for tool in tool_calls:
                        tool_name = (
                            tool.get("name", "")
                            if isinstance(tool, dict)
                            else getattr(tool, "name", "")
                        )
                        tool_args = (
                            tool.get("args", {})
                            if isinstance(tool, dict)
                            else getattr(tool, "args", {})
                        )
                        tool_chain.append(
                            {"type": "tool_call", "name": tool_name, "args": tool_args}
                        )
                else:
                    content = (
                        msg.content[:60] + "..."
                        if len(msg.content) > 60
                        else msg.content
                    )
                    tool_chain.append({"type": "ai_response", "content": content})
            elif msg_type == "ToolMessage":
                content = (
                    str(msg.content)[:80] + "..."
                    if len(str(msg.content)) > 80
                    else str(msg.content)
                )
                tool_chain.append({"type": "tool_result", "content": content})
        return tool_chain

    # -------------------------------------------------------------------------
    # 辅助：解析 Agent 输出（JSON / 纯文本）
    # -------------------------------------------------------------------------
    def _parse_agent_output(
        self, content: str, parser_type: Optional[str], stock: str
    ) -> dict:
        """解析 Agent 输出。json 解析失败时兜底为 HOLD，并打 parse_fallback 标记，
        否则评测无法区分"真的观望"与"输出坏了被兜底"（见 Eval/reliability.py）。"""
        if parser_type == "json" and content:
            try:
                if "```json" in content:
                    json_str = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    json_str = content.split("```")[1].split("```")[0].strip()
                else:
                    json_str = content.strip()

                result = json.loads(json_str)
                if "decisions" in result and len(result["decisions"]) > 0:
                    decision = result["decisions"][0]
                    return {
                        "symbol": decision.get("symbol", stock),
                        "vote": decision.get("vote", "HOLD"),
                        "target_position_pct": decision.get("target_position_pct", 0.5),
                        "confidence": decision.get("confidence", 0.7),
                        "reason": decision.get("reason", ""),
                        "portfolio_suggestion": result.get("portfolio_suggestion", ""),
                    }
            except Exception:
                pass  # fallback 到默认返回

        # 默认 / 纯文本返回；json 解析器走到这里说明输出不合法，标记为降级
        return {
            "symbol": stock,
            "vote": "HOLD",
            "target_position_pct": 0.5,
            "confidence": 0.5,
            "reason": content[:500] if content else "无分析结果",
            "portfolio_suggestion": "",
            "parse_fallback": parser_type == "json",
        }

    # -------------------------------------------------------------------------
    # 多 Agent 并行模式（ALL）— 从注册表遍历，不再硬编码
    # -------------------------------------------------------------------------
    async def _run_all_agents(self, state: GraphStatus) -> dict:
        print(f"\n┌─ [ALL] 多Agent投票模式")

        stock = self._extract_stock(state)

        # 并行执行本次选中的 Agent（selected_agents；缺省 = 全部已装配）
        selected = state.get("selected_agents") or []
        agent_names = [n for n in selected if n in self.agent] or [
            n for n in _registry_all_agents() if n in self.agent
        ]
        if not agent_names:
            print(f"│  警告: 没有可用的 Agent")
            return {"agent_results": {}, "stock": stock, "status": "voting_ready"}

        tasks = [
            asyncio.create_task(
                self._run_single_agent_async(
                    name,
                    stock,
                    state,
                    user_id=state.get("user_id", "anonymous"),
                )
            )
            for name in agent_names
        ]
        # return_exceptions=True：单个 Agent 异常不拖垮整体（配合底层兜底双保险）
        results = await asyncio.gather(*tasks, return_exceptions=True)

        agent_results = {}
        for name, r in zip(agent_names, results):
            if isinstance(r, Exception):
                print(f"│  ❌ {name} 抛异常，已兜底为观望: {str(r)[:80]}")
                agent_results[name] = {
                    "symbol": stock,
                    "vote": "HOLD",
                    "target_position_pct": 0.3,
                    "confidence": 0.3,
                    "reason": f"{name} 分析暂时不可用（{str(r)[:80]}），暂持观望。",
                    "error": True,
                    "tool_chain": [],
                    "raw_output": "",
                    "agent_name": name,
                }
            else:
                agent_results[r["agent_name"]] = r

        for name, r in agent_results.items():
            print(f"│  ✅ {name} 分析完成 -> {r.get('vote', 'HOLD')}")
        print(f"\n│  ✅ 多Agent分析完成，进入投票阶段...")

        return {
            "agent_results": agent_results,
            "stock": stock,
            "status": "voting_ready",
        }

    # -------------------------------------------------------------------------
    # 通用投票汇总 — 遍历 agent_results，支持动态权重
    # -------------------------------------------------------------------------
    async def _voting(self, state: GraphStatus) -> dict:
        print(f"\n┌─ [VOTING] 投票汇总")
        emit_progress({"stage": "voting"})

        agent_results = state.get("agent_results") or {}
        if not agent_results:
            print(f"│  无分析结果")
            return {"result": "无分析结果", "status": "completed"}

        # 取第一个结果的 symbol 作为默认
        first = next(iter(agent_results.values()))
        symbol = first.get("symbol", "UNKNOWN")

        # 获取 Agent 权重（如果 memory_manager 可用）
        agent_names = list(agent_results.keys())
        weights = {name: 1.0 for name in agent_names}
        voter_uid = state.get("user_id") or "anonymous"
        if self.memory_manager:
            try:
                # 与反馈写入口径一致：客观权重(global) × 该用户的个人调整
                loaded = await self.memory_manager.resolve_weights(voter_uid, agent_names)
                weights.update(loaded)
            except Exception as e:
                print(f"│  ⚠️ 权重加载失败: {e}")

        # 加权投票统计
        weighted_buy = 0.0
        weighted_sell = 0.0
        weighted_reduce = 0.0
        weighted_hold = 0.0
        weighted_target_pct = 0.0
        total_weight = 0.0

        print(f"│  股票: {symbol}")
        for name, r in agent_results.items():
            w = weights.get(name, 1.0)
            vote = r.get("vote", "HOLD")
            pct = r.get("target_position_pct", 0.5)
            conf = r.get("confidence", 0.75)
            print(
                f"│  {name}: {vote} "
                f"(仓位{pct*100:.0f}%, 置信度{conf*100:.0f}%, 权重{w:.2f})"
            )
            if vote in ("BUY", "STRONG_BUY"):
                weighted_buy += w
            elif vote in ("SELL", "STRONG_SELL"):
                weighted_sell += w
            elif vote == "REDUCE":
                weighted_reduce += w
            else:
                weighted_hold += w
            weighted_target_pct += pct * w
            total_weight += w

        # 归一化
        norm_buy = weighted_buy / total_weight if total_weight else 0
        norm_sell = weighted_sell / total_weight if total_weight else 0
        norm_reduce = weighted_reduce / total_weight if total_weight else 0
        norm_hold = weighted_hold / total_weight if total_weight else 0
        avg_target_pct = weighted_target_pct / total_weight if total_weight else 0.4

        # 决策阈值（可配置）
        strong_th = _config.vote_strong_threshold if _config else 0.75
        majority_th = _config.vote_majority_threshold if _config else 0.4
        reduce_th = _config.vote_reduce_threshold if _config else 0.4

        # 决策逻辑
        if norm_buy >= strong_th:
            final_vote, confidence, suggestion = (
                "STRONG_BUY",
                int(70 + norm_buy * 30),
                "加权全体看多，建议大胆买入",
            )
            target_pct = min(0.8, avg_target_pct * 1.2)
        elif norm_buy > majority_th and norm_sell == 0 and norm_reduce == 0:
            final_vote, confidence, suggestion = "BUY", int(50 + norm_buy * 40), "加权多数看多，可考虑买入"
            target_pct = avg_target_pct * 0.8
        elif norm_sell >= strong_th:
            final_vote, confidence, suggestion = (
                "STRONG_SELL",
                int(70 + norm_sell * 30),
                "加权全体看空，建议果断卖出",
            )
            target_pct = 0.0
        elif norm_sell > majority_th and norm_buy == 0:
            final_vote, confidence, suggestion = "SELL", int(50 + norm_sell * 40), "加权多数看空，建议减仓"
            target_pct = 0.1
        elif norm_reduce > reduce_th and norm_buy == 0 and norm_sell == 0:
            final_vote, confidence, suggestion = (
                "REDUCE",
                int(40 + norm_reduce * 40),
                "加权风险指标提示减仓，建议降低仓位",
            )
            target_pct = avg_target_pct * 0.5
        else:
            final_vote, confidence, suggestion = "HOLD", int(30 + max(norm_buy, norm_sell, norm_hold) * 40), "加权分歧或观望，建议持有"
            target_pct = max(0.3, min(0.5, avg_target_pct))

        print(
            f"│  📊 最终决策: {final_vote} (置信度: {confidence}%, 目标仓位: {target_pct*100:.0f}%)"
        )

        # 构建各 Agent 详细决策
        decisions = []
        all_tool_chains = []
        result_parts = []

        for agent_name, result in agent_results.items():
            w = weights.get(agent_name, 1.0)
            decisions.append(
                {
                    "symbol": result.get("symbol", symbol),
                    "vote": result.get("vote", "HOLD"),
                    "reason": result.get("reason", ""),
                    "target_position_pct": result.get("target_position_pct", 0.5),
                    "confidence": result.get("confidence", 0.7),
                    "weight": w,
                }
            )
            if result.get("tool_chain"):
                all_tool_chains.append(
                    {"agent": agent_name, "steps": result["tool_chain"]}
                )

            result_parts.append(
                f"### 🤖 {agent_name} (权重: {w:.2f})\n\n"
                f"**投票**: {result.get('vote', 'HOLD')} | "
                f"**建议仓位**: {result.get('target_position_pct', 0.5)*100:.0f}% | "
                f"**置信度**: {result.get('confidence', 0.7)*100:.0f}%\n\n"
                f"**分析理由**:\n{result.get('reason', '暂无分析')}"
            )

        final_decision = {
            "symbol": symbol,
            "final_vote": final_vote,
            "target_position_pct": round(target_pct, 2),
            "confidence": confidence,
            "suggestion": suggestion,
        }

        detailed_analysis = {
            "decisions": decisions,
            "portfolio_suggestion": suggestion,
            "final_decision": {
                "vote": final_vote,
                "confidence": confidence / 100,
                "target_position_pct": target_pct,
                "reason": suggestion,
            },
        }

        result_text = (
            f"## 📊 多Agent综合分析 - {symbol}\n\n"
            f"---\n\n" + "\n\n---\n\n".join(result_parts) + f"\n\n---\n\n"
            f"### 🎯 最终加权投票结果\n\n"
            f"**综合决策**: {final_vote} (置信度: {confidence}%)  \n"
            f"**建议仓位**: {target_pct*100:.0f}%  \n"
            f"**操作建议**: {suggestion}"
        )

        return {
            "result": result_text,
            "final_decision": final_decision,
            "detailed_analysis": detailed_analysis,
            "all_tool_chains": all_tool_chains,
            "stock": symbol,
            "status": "completed",
        }

    # -------------------------------------------------------------------------
    # 澄清节点
    # -------------------------------------------------------------------------
    def _clarify(self, state: GraphStatus) -> dict:
        print(f"\n┌─ [clarify_node] 需要澄清")
        emit_progress({"stage": "clarify"})
        clarification_question = state.get("parse_result", {}).get(
            "clarification",
            "抱歉，我没有理解您的问题。请提供更多关于股票名称或分析需求的信息。",
        )
        count = state.get("clarification_count", 0) + 1
        print(f"│  澄清问题: {clarification_question}")
        print(f"│  澄清次数: {count}/3")
        print(f"└─ 中断: 请用户回复后，用 resume() 继续")
        return {
            "result": clarification_question,
            "status": "waiting_for_clarification",
            "clarification_count": count,
        }

    # -------------------------------------------------------------------------
    # 对外接口
    # -------------------------------------------------------------------------
    async def run(
        self,
        user_input: str,
        thread_id: str = None,
        user_id: Optional[str] = None,
        agents: Optional[list] = None,
        progress_cb=None,
    ) -> dict:
        """Graph 入口。progress_cb: 可选回调 async/sync fn(event: dict)，
        用于 SSE 实时推送执行阶段（preprocess/agent_start/agent_done/voting/chat…）。"""
        token = _progress_cb_var.set(progress_cb)
        try:
            return await self._run_impl(user_input, thread_id, user_id, agents)
        finally:
            _progress_cb_var.reset(token)

    async def _run_impl(
        self,
        user_input: str,
        thread_id: str = None,
        user_id: Optional[str] = None,
        agents: Optional[list] = None,
    ) -> dict:
        thread_id = thread_id or "default_thread"
        config = {"configurable": {"thread_id": thread_id}}

        current_state = await self.graph.aget_state(config)
        if (
            current_state
            and current_state.values.get("status") == "waiting_for_clarification"
        ):
            print(f"\n{'='*60}")
            print(f"🔄 澄清回复后重新运行 | 输入: '{user_input}'")
            print(f"{'='*60}")
            count = current_state.values.get("clarification_count", 0)
        else:
            print(f"\n{'='*60}")
            print(f"🚀 开始运行 Graph | 输入: '{user_input}'")
            print(f"{'='*60}")
            count = 0

        initial_state = {
            "input": user_input,
            "parse_result": None,
            "result": None,
            "status": "init",
            "thread_id": thread_id,
            "next_agent": None,
            "current_agent": None,
            "clarification_count": count,
            "agent_results": None,
            "final_decision": None,
            "selected_agents": agents or None,
            "route_info": None,
            "user_id": user_id or "anonymous",
        }

        try:
            async for event in self.graph.astream(initial_state, config=config):
                pass
        except Exception as e:
            # 顶层兜底：图执行任何异常都转换为结构化错误返回，不向上抛出（不会让 API 500）
            import traceback

            traceback.print_exc()
            print(f"\n❌ Graph 运行失败: {e}")
            return {
                "input": user_input,
                "thread_id": thread_id,
                "status": "error",
                "result": "分析过程出现异常，请换个问法重试一次，或稍后再试。",
                "parse_result": None,
                "agent_results": None,
                "final_decision": None,
            }

        final = await self.graph.aget_state(config)
        final_values = dict(final.values) if final else {}

        if final_values.get("status") == "waiting_for_clarification":
            print(f"\n⏸️ 等待用户澄清...")
            return final_values

        # Passive + Active Self-Improve: record analysis & check consistency
        if final_values.get("status") == "completed" and self.memory_manager:
            try:
                stock = final_values.get("stock", "UNKNOWN")
                agent_results = final_values.get("agent_results", {})
                final_decision = final_values.get("final_decision", {})
                votes = {
                    name: r.get("vote", "HOLD")
                    for name, r in agent_results.items()
                }
                user_id = final_values.get("user_id", "anonymous")
                details = {
                    name: {
                        "target_position_pct": r.get("target_position_pct"),
                        "confidence": r.get("confidence"),
                    }
                    for name, r in agent_results.items()
                }
                record_id = await self.memory_manager.record_analysis(
                    session_id=thread_id,
                    stock_symbol=stock,
                    query=user_input,
                    agent_votes=votes,
                    final_decision=final_decision.get("final_vote", "HOLD"),
                    reasoning_summary=final_decision.get("suggestion", ""),
                    user_id=user_id,
                    agent_details=details,
                )
                # Active: check for inconsistencies with recent history（排除本次记录）
                inconsistency = await self.memory_manager.find_inconsistencies(
                    user_id=user_id,
                    stock_symbol=stock,
                    current_decision=final_decision.get("final_vote", "HOLD"),
                    exclude_id=record_id,
                )
                if inconsistency:
                    print(f"│  ⚠️ Self-Improve: {inconsistency}")
            except Exception as e:
                print(f"│  ⚠️ L3 record failed: {e}")

        print(f"\n{'='*60}")
        print(f"✅ Graph 运行完成")
        print(f"{'='*60}")
        return final_values

    async def resume(
        self,
        user_input: str,
        thread_id: str,
        user_id: Optional[str] = None,
    ) -> dict:
        print(f"\n🔄 [resume] 用户回复澄清 | thread: {thread_id}")
        print(f"   用户回复: '{user_input}'")
        return await self.run(user_input, thread_id, user_id=user_id)


# =============================================================================
# 测试脚本
# =============================================================================


def _create_test_agents(checkpointer, memory_manager=None):
    """统一构造测试 Agent 实例（注册表装配，含 DIY agent）"""
    from LLM.registry import all_agents as _registry_all_agents, instantiate as _instantiate
    from LLM.llm import create_chat_model

    model = create_chat_model(temperature=0.3)
    preprocessor = Preprocessor(model, checkpointer)
    router = Router()

    agents = {
        name: _instantiate(name, model, checkpointer, memory_manager)
        for name in _registry_all_agents()
    }

    fin_graph = FinGraph(
        preprocessor=preprocessor,
        router=router,
        agent=agents,
        checkpointer=checkpointer,
        memory_manager=memory_manager,
    )
    return fin_graph


def test_with_clarification_flow():
    print("\n" + "=" * 60)
    print("测试场景: 需要澄清的对话流程")
    print("=" * 60)
    print("\n💡 流程说明:")
    print("   1. 用户输入模糊问题 -> 系统要求澄清")
    print("   2. 用户补充信息 -> 系统继续处理")
    print("   3. 最终返回分析结果")

    checkpointer = MemorySaver()
    fin_graph = _create_test_agents(checkpointer)
    thread_id = "test_clarify_001"

    # 第 1 轮：模糊输入触发澄清
    print("\n" + "-" * 60)
    print("[第1轮] 用户输入: '分析一下这个股票'")
    print("-" * 60)
    result1 = asyncio.run(fin_graph.run("分析一下这个股票", thread_id))

    if result1.get("status") == "waiting_for_clarification":
        # 第 2 轮：用户澄清回复
        print("\n" + "-" * 60)
        print("[第2轮] 用户看到系统提示后回复: '茅台'")
        print("-" * 60)
        result2 = asyncio.run(fin_graph.resume("贵州茅台", thread_id))

        print("\n" + "-" * 60)
        print("📋 最终结果:")
        print("-" * 60)
        print(f"   状态: {result2.get('status')}")
        print(f"   结果: {result2.get('result')}")
    else:
        print("\n⚠️ 注意: 没有触发澄清，直接完成了")
        print(f"   状态: {result1.get('status')}")
        print(f"   结果: {result1.get('result')}")


def test_normal_flow():
    print("\n" + "=" * 60)
    print("测试场景: 正常直接完成的流程")
    print("=" * 60)

    checkpointer = MemorySaver()
    fin_graph = _create_test_agents(checkpointer)

    # 打印图结构
    print("\n📊 Graph 结构 (Mermaid):")
    print(fin_graph.graph.get_graph().draw_mermaid())

    test_cases = [
        ("分析一下茅台的K线走势", "thread_normal_1"),
        ("寒武纪股票估值高吗？建议买入吗？", "thread_normal_3"),
    ]

    for user_input, thread_id in test_cases:
        print("\n" + "=" * 60)
        result = asyncio.run(fin_graph.run(user_input, thread_id))
        print(f"\n📋 最终结果:")
        print(f"   status: {result.get('status')}")
        r = result.get("result")
        print(f"   result: {r[:50]}..." if r and len(r) > 50 else f"   result: {r}")


def test_all_agents_flow():
    print("\n" + "=" * 60)
    print("测试场景: 双Agent投票模式 (ALL)")
    print("=" * 60)

    checkpointer = MemorySaver()
    fin_graph = _create_test_agents(checkpointer)
    thread_id = "test_all_001"

    # 通过覆盖路由让它走 ALL 模式，或者直接测试 Router 返回 ALL 的 intent
    # 这里我们手动构造一个 ALL 路由的场景
    print("\n" + "-" * 60)
    print("[测试] 用户输入: '给我投资建议'")
    print("-" * 60)
    result = asyncio.run(fin_graph.run("给我投资建议", thread_id))

    print(f"\n📋 最终结果:")
    print(f"   status: {result.get('status')}")
    r = result.get("result")
    print(f"   result: {r[:80]}..." if r and len(r) > 80 else f"   result: {r}")


if __name__ == "__main__":
    print("=" * 60)
    print("FinGraph 测试脚本（配置驱动版本）")
    print("=" * 60)

    test_normal_flow()
    test_with_clarification_flow()
    test_all_agents_flow()

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
