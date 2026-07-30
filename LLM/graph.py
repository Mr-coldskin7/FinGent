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
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
import json
import asyncio

load_dotenv()
qianwen_api_key = os.getenv("QIANWEN_API_KEY")


# =============================================================================
# Agent 注册表 — 加新 Agent 只需要在这里添加配置
# =============================================================================
AGENT_CONFIG: Dict[str, dict] = {
    "Morefit": {
        "output_parser": "json",
        "description": "基本面分析 Agent",
    },
    "TECHNICAL_NERD": {
        "output_parser": "json",
        "description": "技术面分析 Agent",
    },
    "RiskManager": {
        "output_parser": "json",
        "description": "风险与仓位控制 Agent",
    },
    "SentimentAnalyzer": {
        "output_parser": "json",
        "description": "舆情分析 Agent",
    },
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

        workflow.set_entry_point("preprocess")
        workflow.add_edge("preprocess", "route")

        # 动态路由映射：从注册表自动生成 + 内置节点
        route_mapping = {name: "execute_agent" for name in AGENT_CONFIG}
        route_mapping["ALL"] = "ALL"
        route_mapping["clarify_node"] = "clarify_node"
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

        return workflow.compile(checkpointer=self.checkpointer)

    # -------------------------------------------------------------------------
    # 预处理节点
    # -------------------------------------------------------------------------
    async def _preprocess(self, state: GraphStatus) -> dict:
        print(f"\n┌─ [preprocess] 预处理节点")
        print(f"│  输入: '{state['input']}'")
        result = await self.preprocessor.async_parse(
            state["input"], thread_id=state["thread_id"]
        )
        print(
            f"│  解析完成: intent={result.get('intent')}, stock={result.get('stock')}"
        )
        print(f"└─ 输出: status={result['status']}")
        return {"parse_result": result, "status": result["status"]}

    # -------------------------------------------------------------------------
    # 路由节点
    # -------------------------------------------------------------------------
    def _route(self, state: GraphStatus) -> dict:
        print(f"\n┌─ [route] 路由节点")
        parse_result = state["parse_result"]
        print(f"│  parse_result: {parse_result}")

        # 1. 需要澄清
        if parse_result.get("status") == "clarification_needed":
            print(f"└─ 路由到: -> clarify_node")
            return {"next_agent": "clarify_node", "status": "need_clarification"}

        # 2. 缺少实体
        entities = parse_result.get("entities", {})
        codes = entities.get("code", [])
        symbols = entities.get("symbols", [])
        names = entities.get("names", [])

        if not codes and not symbols and not names:
            print(f"│  未识别到公司实体")
            print(f"└─ 路由到: -> clarify_node")
            return {"next_agent": "clarify_node", "status": "need_stock_name"}

        # 3. 正常路由 — 同时设置 current_agent 供 execute_agent 使用
        result = self.router.route(parse_result)
        stock = (
            codes[0]
            if codes
            else symbols[0] if symbols else names[0] if names else "未知"
        )
        print(f"│  公司: {stock}")
        print(f"└─ 路由到: -> {result}")
        return {
            "next_agent": result,
            "current_agent": result,
            "status": "routed",
            "stock": stock,
        }

    # -------------------------------------------------------------------------
    # 统一单 Agent 执行节点（所有单 Agent 模式走这里）
    # -------------------------------------------------------------------------
    async def _execute_agent(self, state: GraphStatus) -> dict:
        agent_name = state.get("current_agent")
        print(f"\n┌─ [execute_agent] 统一执行节点 | Agent: {agent_name}")

        if not agent_name or agent_name not in AGENT_CONFIG:
            print(f"│  错误: Agent '{agent_name}' 未在配置表中定义")
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
        config = AGENT_CONFIG[agent_name]
        if config.get("output_parser") == "json":
            result_text = (
                f"## 📊 {agent_name} 分析 - {result['symbol']}\n\n"
                f"**投票**: {result['vote']} | "
                f"**建议仓位**: {result['target_position_pct']*100:.0f}% | "
                f"**置信度**: {result['confidence']*100:.0f}%\n\n"
                f"**分析理由**:\n{result['reason']}"
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
        config = AGENT_CONFIG.get(agent_name, {})

        if not agent:
            return {
                "symbol": stock,
                "vote": "HOLD",
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

        # 执行 Agent（传入 stock 和 user_id 以触发 L3 记忆注入）
        response = await agent.achat(
            prompt,
            thread_id=agent_thread_id,
            stock=stock,
            user_id=user_id,
        )

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

        return parsed

    # -------------------------------------------------------------------------
    # 辅助：从 State 提取股票信息
    # -------------------------------------------------------------------------
    def _extract_stock(self, state: GraphStatus) -> str:
        entities = state.get("parse_result", {}).get("entities", {})
        codes = entities.get("code", [])
        symbols = entities.get("symbols", [])
        names = entities.get("names", [])
        return (
            codes[0]
            if codes
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

        # 默认 / 纯文本返回
        return {
            "symbol": stock,
            "vote": "HOLD",
            "target_position_pct": 0.5,
            "confidence": 0.5,
            "reason": content[:500] if content else "无分析结果",
            "portfolio_suggestion": "",
        }

    # -------------------------------------------------------------------------
    # 多 Agent 并行模式（ALL）— 从注册表遍历，不再硬编码
    # -------------------------------------------------------------------------
    async def _run_all_agents(self, state: GraphStatus) -> dict:
        print(f"\n┌─ [ALL] 多Agent投票模式")

        stock = self._extract_stock(state)

        # 并行执行所有已注册且已初始化的 Agent
        agent_names = [name for name in AGENT_CONFIG if name in self.agent]
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
        results = await asyncio.gather(*tasks)

        agent_results = {r["agent_name"]: r for r in results}

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
        if self.memory_manager:
            try:
                loaded = await self.memory_manager.get_agent_weights(
                    user_id="global", agent_names=agent_names
                )
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
            "user_id": user_id or "anonymous",
        }

        async for event in self.graph.astream(initial_state, config=config):
            pass

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
                await self.memory_manager.record_analysis(
                    session_id=thread_id,
                    stock_symbol=stock,
                    query=user_input,
                    agent_votes=votes,
                    final_decision=final_decision.get("final_vote", "HOLD"),
                    reasoning_summary=final_decision.get("suggestion", ""),
                    user_id=user_id,
                )
                # Active: check for inconsistencies with recent history
                inconsistency = await self.memory_manager.find_inconsistencies(
                    user_id=user_id,
                    stock_symbol=stock,
                    current_decision=final_decision.get("final_vote", "HOLD"),
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
    """统一构造测试 Agent 实例"""
    from LLM.agent import RiskManager, SentimentAnalyzer
    from LLM.risk_tools import RISK_TOOLS
    from LLM.sentiment_tools import SENTIMENT_TOOLS

    model = ChatOpenAI(
        api_key=qianwen_api_key,
        base_url=os.getenv(
            "MODEL_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
        ),
        model="qwen-max",
        temperature=0.3,
    )
    preprocessor = Preprocessor(model, checkpointer)
    router = Router()

    agents = {
        "TECHNICAL_NERD": TECHNICAL_NERD(
            model=model,
            tools=[get_stock_price, get_stock_basic_info],
            checkpointer=checkpointer,
            memory_manager=memory_manager,
        ),
        "Morefit": Morefit(
            model=model,
            tools=[
                get_stock_company_info,
                get_stock_financial_report_links,
                get_stock_price,
                get_stock_basic_info,
            ],
            checkpointer=checkpointer,
            memory_manager=memory_manager,
        ),
        "RiskManager": RiskManager(
            model=model, tools=RISK_TOOLS, checkpointer=checkpointer, memory_manager=memory_manager
        ),
        "SentimentAnalyzer": SentimentAnalyzer(
            model=model, tools=SENTIMENT_TOOLS, checkpointer=checkpointer, memory_manager=memory_manager
        ),
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
