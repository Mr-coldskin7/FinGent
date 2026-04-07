from typing import TypedDict, Any, Optional, List, Dict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
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
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_community.chat_models import ChatTongyi
import os
from dotenv import load_dotenv
import json
import asyncio
load_dotenv()
qianwen_api_key = os.getenv("QIANWEN_API_KEY")

class GraphStatus(TypedDict):
    input: str
    parse_result: Optional[dict]
    result: Any
    status: str
    thread_id: str
    next_agent: Optional[str]
    clarification_count: int  # 记录澄清次数，防止无限循环
    morefit_vote: Optional[dict]  # Morefit的投票结果
    tech_vote: Optional[dict]     # TECHNICAL_NERD的投票结果
    final_decision: Optional[dict]  # 最终汇总决策
    all_tool_chains: Optional[list]  # 双Agent模式的工具调用链
    detailed_analysis: Optional[dict]  # 详细分析数据
    tool_chain: Optional[list]  # 单Agent模式的工具调用链
    agent_name: Optional[str]  # Agent名称
    stock: Optional[str]  # 股票代码


class FinGraph:
    def __init__(self, preprocessor: Preprocessor, router: Router, agent: dict = None, checkpointer=None):
        self.preprocessor = preprocessor
        self.router = router
        self.agent = agent or {}
        # 如果没有提供 checkpointer，使用 MemorySaver
        if checkpointer is None:
            self.checkpointer = MemorySaver()
        else:
            self.checkpointer = checkpointer
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(GraphStatus)
        
        # 添加节点
        workflow.add_node("preprocess", self._preprocess)
        workflow.add_node("route", self._route)
        workflow.add_node("TECHNICAL_NERD", self._Technical_nerd)
        workflow.add_node("Morefit", self._Morefit)
        workflow.add_node("clarify_node", self._clarify)
        
        # 设置入口
        workflow.set_entry_point("preprocess")
        
        # preprocess -> route (普通边)
        workflow.add_edge("preprocess", "route")
        
        # route -> agent/clarify (条件边)
        workflow.add_conditional_edges(
            "route",
            lambda state: state.get("next_agent", "end"),
            {
                "TECHNICAL_NERD": "TECHNICAL_NERD",
                "Morefit": "Morefit",
                "ALL": "ALL",  # 新增：双Agent投票模式
                "clarify_node": "clarify_node",
                "end": END
            }
        )
        
        # 添加双Agent投票节点
        workflow.add_node("ALL", self._run_all_agents)
        workflow.add_node("voting", self._voting)  # 投票汇总节点
        
        # agent -> END (普通单Agent模式)
        workflow.add_edge("TECHNICAL_NERD", END)
        workflow.add_edge("Morefit", END)
        
        # ALL模式: 并行执行两个Agent -> 投票汇总 -> END
        workflow.add_edge("ALL", "voting")
        workflow.add_edge("voting", END)
        
        # clarify -> 结束（等待用户回复后从外部重新启动）
        workflow.add_edge("clarify_node", END)
        
        return workflow.compile(checkpointer=self.checkpointer)

    def _preprocess(self, state: GraphStatus) -> dict:
        """预处理节点 - 解析用户输入"""
        print(f"\n┌─ [preprocess] 预处理节点")
        print(f"│  输入: '{state['input']}'")
        result = self.preprocessor.parse(
            state["input"], 
            thread_id=state["thread_id"]
        )
        print(f"│  解析完成: intent={result.get('intent')}, stock={result.get('stock')}")
        print(f"└─ 输出: status={result['status']}")
        return {
            "parse_result": result,
            "status": result["status"]
        }

    def _route(self, state: GraphStatus) -> dict:
        """路由节点 - 决定走哪个 agent"""
        print(f"\n┌─ [route] 路由节点")
        
        parse_result = state["parse_result"]
        print(f"│  parse_result: {parse_result}")
        
        # 1. 检查状态是否需要澄清
        if parse_result.get("status") == "clarification_needed":
            print(f"└─ 路由到: -> clarify_node (状态标记需要澄清)")
            return {"next_agent": "clarify_node", "status": "need_clarification"}
        
        # 2. 检查是否有实体信息 (code / symbols / names)
        entities = parse_result.get("entities", {})
        codes = entities.get("code", [])
        symbols = entities.get("symbols", [])
        names = entities.get("names", [])
        
        if not codes and not symbols and not names:
            print(f"│  ⚠️ 未识别到公司实体")
            print(f"└─ 路由到: -> clarify_node (需要用户补充股票名称)")
            return {"next_agent": "clarify_node", "status": "need_stock_name"}
        
        # 3. 正常路由到 Agent
        result = self.router.route(parse_result)
        # 优先级: code > symbols > names
        stock = codes[0] if codes else symbols[0] if symbols else names[0] if names else "未知"
        print(f"│  公司: {stock}")
        print(f"└─ 路由到: -> {result}")
        return {"next_agent": result, "status": "routed"}
    
    def _Morefit(self, state: GraphStatus) -> dict:
        """Morefit Agent - 基本面分析与投资建议"""
        print(f"\n┌─ [Morefit] 基本面分析 Agent")
        
        # 从 entities 中获取股票信息 (优先级: code > symbols > names)
        entities = state.get('parse_result', {}).get('entities', {})
        codes = entities.get('code', [])
        symbols = entities.get('symbols', [])
        names = entities.get('names', [])
        stock = codes[0] if codes else symbols[0] if symbols else names[0] if names else "未知股票"
        
        print(f"│  分析股票: {stock}")
        print(f"│  用户问题: '{state['input']}'")
        print(f"│  调用 Morefit Agent 分析中...")
        
        # 调用真实的 Morefit Agent
        morefit_agent = self.agent.get('Morefit')
        tool_chain = []  # 收集工具调用链
        
        if morefit_agent:
            # 构造提示词 - 强制只分析一只股票，避免模型发散
            prompt = f"""请分析股票：{stock}

要求：
1. 只分析这一只股票，不要查询其他股票
2. 用户问题：{state['input']}
3. 使用工具获取数据后，输出纯JSON格式分析结果"""
            response = morefit_agent.chat(prompt, thread_id=state['thread_id'])
            analysis = response.final_answer
            
            # 收集工具调用链
            print(f"│  📋 工具调用链 (共 {response.step_count} 步, {response.tool_calls} 个工具):")
            print(f"│  📨 消息数量: {len(response.messages)}")
            for i, msg in enumerate(response.messages):
                msg_type = type(msg).__name__
                has_tool_calls = hasattr(msg, 'tool_calls') and msg.tool_calls
                print(f"│    消息[{i}]: {msg_type} | tool_calls: {has_tool_calls}")
                if has_tool_calls:
                    print(f"│      工具: {[t.get('name') if isinstance(t, dict) else getattr(t, 'name', '') for t in msg.tool_calls]}")
                
                # 使用类型名称检查
                if msg_type == 'HumanMessage':
                    content = msg.content[:50] + "..." if len(msg.content) > 50 else msg.content
                    print(f"│     👤 输入: {content}")
                    tool_chain.append({"type": "input", "content": content})
                elif msg_type == 'AIMessage' and msg.tool_calls:
                    for tool in msg.tool_calls:
                        tool_name = tool.get('name', '') if isinstance(tool, dict) else getattr(tool, 'name', '')
                        tool_args = tool.get('args', {}) if isinstance(tool, dict) else getattr(tool, 'args', {})
                        print(f"│     🔧 调用: {tool_name}({tool_args})")
                        tool_chain.append({
                            "type": "tool_call", 
                            "name": tool_name, 
                            "args": tool_args
                        })
                elif msg_type == 'ToolMessage':
                    content = str(msg.content)[:100] + "..." if len(str(msg.content)) > 100 else str(msg.content)
                    print(f"│     📊 结果: {content}")
                    tool_chain.append({"type": "tool_result", "content": content})
            print(f"│  ✓ Agent 分析完成")
        else:
            # Fallback: 如果没有 agent 实例，使用模拟结果
            analysis = f"【Morefit 分析】{stock}: PE合理，建议中长期持有 (模拟结果)"
            print(f"│  ⚠️ 未找到 Morefit Agent 实例，使用模拟结果")
        
        return {
            "result": analysis, 
            "status": "completed",
            "agent_name": "Morefit",
            "tool_chain": tool_chain,
            "stock": stock
        }
    
    def _Technical_nerd(self, state: GraphStatus) -> dict:
        """Technical_nerd Agent - 技术面分析"""
        print(f"\n┌─ [Technical_nerd] 技术分析 Agent")
        
        # 从 entities 中获取股票信息 (优先级: code > symbols > names)
        entities = state.get('parse_result', {}).get('entities', {})
        codes = entities.get('code', [])
        symbols = entities.get('symbols', [])
        names = entities.get('names', [])
        stock = codes[0] if codes else symbols[0] if symbols else names[0] if names else "未知股票"
        
        print(f"│  分析股票: {stock}")
        print(f"│  用户问题: '{state['input']}'")
        print(f"└─ 调用 TECHNICAL_NERD Agent 分析中...")
        
        # 调用真实的 TECHNICAL_NERD Agent
        tech_agent = self.agent.get('TECHNICAL_NERD')
        tool_chain = []  # 收集工具调用链
        
        if tech_agent:
            # 构造提示词，包含股票信息和用户原始问题
            prompt = f"请分析股票 {stock} 的技术面情况。用户原始问题：{state['input']}"
            response = tech_agent.chat(prompt, thread_id=state['thread_id'])
            analysis = response.final_answer
            
            # 收集工具调用链
            print(f"│  📋 工具调用链 (共 {response.step_count} 步, {response.tool_calls} 个工具):")
            for msg in response.messages:
                msg_type = type(msg).__name__
                if msg_type == 'HumanMessage':
                    content = msg.content[:50] + "..." if len(msg.content) > 50 else msg.content
                    print(f"│     👤 输入: {content}")
                    tool_chain.append({"type": "input", "content": content})
                elif msg_type == 'AIMessage' and msg.tool_calls:
                    for tool in msg.tool_calls:
                        tool_name = tool.get('name', '') if isinstance(tool, dict) else getattr(tool, 'name', '')
                        tool_args = tool.get('args', {}) if isinstance(tool, dict) else getattr(tool, 'args', {})
                        print(f"│     🔧 调用: {tool_name}({tool_args})")
                        tool_chain.append({
                            "type": "tool_call", 
                            "name": tool_name, 
                            "args": tool_args
                        })
                elif msg_type == 'ToolMessage':
                    content = str(msg.content)[:100] + "..." if len(str(msg.content)) > 100 else str(msg.content)
                    print(f"│     📊 结果: {content}")
                    tool_chain.append({"type": "tool_result", "content": content})
            print(f"│  ✓ Agent 分析完成")
        else:
            # Fallback: 如果没有 agent 实例，使用模拟结果
            analysis = f"【Technical_nerd 分析】{stock}: MACD金叉，RSI中性偏强 (模拟结果)"
            print(f"│  ⚠️ 未找到 TECHNICAL_NERD Agent 实例，使用模拟结果")
        
        return {
            "result": analysis, 
            "status": "completed",
            "agent_name": "TECHNICAL_NERD",
            "tool_chain": tool_chain,
            "stock": stock
        }
    
    async def _run_all_agents(self, state: GraphStatus) -> dict:
        """双Agent模式 - 分别完整执行Morefit和TECHNICAL_NERD，然后投票"""
        print(f"\n┌─ [ALL] 双Agent投票模式")
        
        # 获取股票信息
        entities = state.get('parse_result', {}).get('entities', {})
        codes = entities.get('code', [])
        symbols = entities.get('symbols', [])
        names = entities.get('names', [])
        stock = codes[0] if codes else symbols[0] if symbols else names[0] if names else "未知股票"
        
        # 并行执行两个Agent（使用 asyncio.gather）
        task_morefit = asyncio.create_task(
            self._run_agent_full_async('Morefit', stock, state)
        )
        task_tech = asyncio.create_task(
            self._run_agent_full_async('TECHNICAL_NERD', stock, state)
        )
        morefit_result, tech_result = await asyncio.gather(task_morefit, task_tech)
        
        print(f"│" + "─"*50)
        print(f"│  ✅ Morefit 分析完成 -> {morefit_result.get('vote', 'HOLD')}")
        print(f"│  ✅ TECHNICAL_NERD 分析完成 -> {tech_result.get('vote', 'HOLD')}")
        print(f"\n│  ✅ 双Agent分析完成，进入投票阶段...")
        return {
            "morefit_vote": morefit_result,
            "tech_vote": tech_result,
            "stock": stock,
            "status": "voting_ready"
        }
    
    def _run_agent_full(self, agent_name: str, stock: str, state: GraphStatus) -> dict:
        """完整执行单个Agent，返回详细信息（包含工具调用链）"""
        agent = self.agent.get(agent_name)
        if not agent:
            return {
                "symbol": stock, 
                "vote": "HOLD", 
                "reason": f"{agent_name}未初始化", 
                "error": True, 
                "tool_chain": [],
                "raw_output": ""
            }
        
        # 构造提示词
        prompt = state['input']
        agent_thread_id = f"{state['thread_id']}_{agent_name}_{id(state)}"
        
        # 执行Agent
        response = agent.chat(prompt, thread_id=agent_thread_id)
        
        # 收集工具调用链（详细记录）
        tool_chain = []
        print(f"│  📋 {agent_name} 工具调用链:")
        print(f"│  📨 消息总数: {len(response.messages)}")
        for i, msg in enumerate(response.messages):
            msg_type_name = type(msg).__name__
            print(f"│    消息[{i}]: 类型={msg_type_name}")
            
            # 使用类型名称检查，避免跨模块isinstance问题
            if msg_type_name == 'HumanMessage':
                content = msg.content[:60] + "..." if len(msg.content) > 60 else msg.content
                print(f"│     👤 输入: {content}")
                tool_chain.append({"type": "input", "content": content})
            elif msg_type_name == 'AIMessage':
                tool_calls = getattr(msg, 'tool_calls', None)
                if tool_calls:
                    for tool in tool_calls:
                        tool_name = tool.get('name', '') if isinstance(tool, dict) else getattr(tool, 'name', '')
                        tool_args = tool.get('args', {}) if isinstance(tool, dict) else getattr(tool, 'args', {})
                        print(f"│     🔧 调用: {tool_name}({tool_args})")
                        tool_chain.append({
                            "type": "tool_call", 
                            "name": tool_name, 
                            "args": tool_args
                        })
                else:
                    content = msg.content[:60] + "..." if len(msg.content) > 60 else msg.content
                    print(f"│     💭 AI: {content}")
            elif msg_type_name == 'ToolMessage':
                content = str(msg.content)[:80] + "..." if len(str(msg.content)) > 80 else str(msg.content)
                print(f"│     📊 结果: {content}")
                tool_chain.append({"type": "tool_result", "content": content})
        print(f"│  📊 共 {len(tool_chain)} 步, {sum(1 for t in tool_chain if t['type'] == 'tool_call')} 个工具调用")
        
        # 解析JSON结果
        try:
            content = response.final_answer
            if '```json' in content:
                json_str = content.split('```json')[1].split('```')[0].strip()
            elif '```' in content:
                json_str = content.split('```')[1].split('```')[0].strip()
            else:
                json_str = content.strip()
            
            result = json.loads(json_str)
            if 'decisions' in result and len(result['decisions']) > 0:
                decision = result['decisions'][0]
                return {
                    "symbol": decision.get('symbol', stock),
                    "vote": decision.get('vote', 'HOLD'),
                    "target_position_pct": decision.get('target_position_pct', 0.5),
                    "confidence": decision.get('confidence', 0.7),
                    "reason": decision.get('reason', ''),
                    "portfolio_suggestion": result.get('portfolio_suggestion', ''),
                    "tool_chain": tool_chain,
                    "agent_name": agent_name,
                    "raw_output": response.final_answer
                }
            else:
                return {
                    "symbol": stock, 
                    "vote": "HOLD", 
                    "target_position_pct": 0.5,
                    "confidence": 0.5,
                    "reason": "无法解析决策", 
                    "raw": content, 
                    "tool_chain": tool_chain, 
                    "agent_name": agent_name,
                    "raw_output": response.final_answer
                }
        except Exception as e:
            return {
                "symbol": stock, 
                "vote": "HOLD", 
                "target_position_pct": 0.5,
                "confidence": 0.5,
                "reason": f"解析错误: {str(e)}", 
                "raw": response.final_answer, 
                "tool_chain": tool_chain, 
                "agent_name": agent_name,
                "raw_output": response.final_answer
            }
    
    async def _run_agent_full_async(self, agent_name: str, stock: str, state: GraphStatus) -> dict:
        """完整执行单个Agent，返回详细信息（包含工具调用链）"""
        agent = self.agent.get(agent_name)
        if not agent:
            return {
                "symbol": stock, 
                "vote": "HOLD", 
                "reason": f"{agent_name}未初始化", 
                "error": True, 
                "tool_chain": [],
                "raw_output": ""
            }
        
        # 构造提示词
        prompt = state['input']
        agent_thread_id = f"{state['thread_id']}_{agent_name}_{id(state)}"
        
        # 执行Agent
        response = await agent.achat(prompt, thread_id=agent_thread_id)
        
        # 收集工具调用链（详细记录）
        tool_chain = []
        print(f"│  📋 {agent_name} 工具调用链:")
        print(f"│  📨 消息总数: {len(response.messages)}")
        for i, msg in enumerate(response.messages):
            msg_type_name = type(msg).__name__
            print(f"│    消息[{i}]: 类型={msg_type_name}")
            
            # 使用类型名称检查，避免跨模块isinstance问题
            if msg_type_name == 'HumanMessage':
                content = msg.content[:60] + "..." if len(msg.content) > 60 else msg.content
                print(f"│     👤 输入: {content}")
                tool_chain.append({"type": "input", "content": content})
            elif msg_type_name == 'AIMessage':
                tool_calls = getattr(msg, 'tool_calls', None)
                if tool_calls:
                    for tool in tool_calls:
                        tool_name = tool.get('name', '') if isinstance(tool, dict) else getattr(tool, 'name', '')
                        tool_args = tool.get('args', {}) if isinstance(tool, dict) else getattr(tool, 'args', {})
                        print(f"│     🔧 调用: {tool_name}({tool_args})")
                        tool_chain.append({
                            "type": "tool_call", 
                            "name": tool_name, 
                            "args": tool_args
                        })
                else:
                    content = msg.content[:60] + "..." if len(msg.content) > 60 else msg.content
                    print(f"│     💭 AI: {content}")
            elif msg_type_name == 'ToolMessage':
                content = str(msg.content)[:80] + "..." if len(str(msg.content)) > 80 else str(msg.content)
                print(f"│     📊 结果: {content}")
                tool_chain.append({"type": "tool_result", "content": content})
        print(f"│  📊 共 {len(tool_chain)} 步, {sum(1 for t in tool_chain if t['type'] == 'tool_call')} 个工具调用")
        
        # 解析JSON结果
        try:
            content = response.final_answer
            if '```json' in content:
                json_str = content.split('```json')[1].split('```')[0].strip()
            elif '```' in content:
                json_str = content.split('```')[1].split('```')[0].strip()
            else:
                json_str = content.strip()
            
            result = json.loads(json_str)
            if 'decisions' in result and len(result['decisions']) > 0:
                decision = result['decisions'][0]
                return {
                    "symbol": decision.get('symbol', stock),
                    "vote": decision.get('vote', 'HOLD'),
                    "target_position_pct": decision.get('target_position_pct', 0.5),
                    "confidence": decision.get('confidence', 0.7),
                    "reason": decision.get('reason', ''),
                    "portfolio_suggestion": result.get('portfolio_suggestion', ''),
                    "tool_chain": tool_chain,
                    "agent_name": agent_name,
                    "raw_output": response.final_answer
                }
            else:
                return {
                    "symbol": stock, 
                    "vote": "HOLD", 
                    "target_position_pct": 0.5,
                    "confidence": 0.5,
                    "reason": "无法解析决策", 
                    "raw": content, 
                    "tool_chain": tool_chain, 
                    "agent_name": agent_name,
                    "raw_output": response.final_answer
                }
        except Exception as e:
            return {
                "symbol": stock, 
                "vote": "HOLD", 
                "target_position_pct": 0.5,
                "confidence": 0.5,
                "reason": f"解析错误: {str(e)}", 
                "raw": response.final_answer, 
                "tool_chain": tool_chain, 
                "agent_name": agent_name,
                "raw_output": response.final_answer
            }
    def _voting(self, state: GraphStatus) -> dict:
        """投票汇总节点 - 综合两个Agent的完整分析结果"""
        print(f"\n┌─ [VOTING] 投票汇总")
        
        # 获取两个Agent的完整结果（与 GraphStatus 定义一致）
        morefit_result = state.get('morefit_vote') or {}
        tech_result = state.get('tech_vote') or {}
        
        symbol = morefit_result.get('symbol', tech_result.get('symbol', 'UNKNOWN'))
        mf_vote = morefit_result.get('vote', 'HOLD')
        tech_vote_val = tech_result.get('vote', 'HOLD')
        mf_target_pct = morefit_result.get('target_position_pct', 0.5)
        tech_target_pct = tech_result.get('target_position_pct', 0.5)
        mf_confidence = morefit_result.get('confidence', 0.75)
        tech_confidence = tech_result.get('confidence', 0.75)
        
        print(f"│  股票: {symbol}")
        print(f"│  Morefit: {mf_vote} (仓位{mf_target_pct*100:.0f}%, 置信度{mf_confidence*100:.0f}%)")
        print(f"│  Tech: {tech_vote_val} (仓位{tech_target_pct*100:.0f}%, 置信度{tech_confidence*100:.0f}%)")
        
        # 投票逻辑
        votes = [mf_vote, tech_vote_val]
        buy_count = votes.count('BUY')
        sell_count = votes.count('SELL')
        
        if buy_count == 2:
            final_vote, confidence, suggestion = 'STRONG_BUY', 90, "双买信号强烈，建议大胆买入"
            target_position_pct = min(0.8, (mf_target_pct + tech_target_pct) / 2 * 1.2)
        elif buy_count == 1 and sell_count == 0:
            final_vote, confidence, suggestion = 'BUY', 70, "基本面或技术面有一方看好，可考虑买入"
            target_position_pct = (mf_target_pct if mf_vote == 'BUY' else tech_target_pct) * 0.6
        elif sell_count == 2:
            final_vote, confidence, suggestion = 'STRONG_SELL', 90, "双卖信号强烈，建议果断卖出"
            target_position_pct = 0.0
        elif sell_count == 1 and buy_count == 0:
            final_vote, confidence, suggestion = 'SELL', 70, "有一方看空，建议减仓"
            target_position_pct = 0.1
        else:
            final_vote, confidence, suggestion = 'HOLD', 50, "双方分歧或均持观望态度，建议持有观望"
            target_position_pct = max(0.3, min(0.5, (mf_target_pct + tech_target_pct) / 2))
        
        print(f"│  📊 最终决策: {final_vote} (置信度: {confidence}%, 目标仓位: {target_position_pct*100:.0f}%)")
        
        # 构建工具调用链列表
        all_tool_chains = []
        if morefit_result.get('tool_chain'):
            all_tool_chains.append({"agent": "Morefit", "steps": morefit_result['tool_chain']})
        if tech_result.get('tool_chain'):
            all_tool_chains.append({"agent": "TECHNICAL_NERD", "steps": tech_result['tool_chain']})
        
        # 构建两个Agent的详细决策
        morefit_decision = {
            "symbol": symbol, "vote": mf_vote, "reason": morefit_result.get('reason', ''),
            "target_position_pct": mf_target_pct, "confidence": mf_confidence
        }
        tech_decision = {
            "symbol": symbol, "vote": tech_vote_val, "reason": tech_result.get('reason', ''),
            "target_position_pct": tech_target_pct, "confidence": tech_confidence
        }
        
        # 最终决策数据
        final_decision = {
            "symbol": symbol, "final_vote": final_vote,
            "target_position_pct": round(target_position_pct, 2),
            "confidence": confidence, "suggestion": suggestion
        }
        
        # 详细分析数据（用于前端展示）
        detailed_analysis = {
            "decisions": [morefit_decision, tech_decision],
            "portfolio_suggestion": suggestion,
            "final_decision": {
                "vote": final_vote, "confidence": confidence / 100,
                "target_position_pct": target_position_pct, "reason": suggestion
            }
        }
        
        # 文本结果 - 包含详细的分析理由
        mf_reason = morefit_result.get('reason', '暂无分析')
        tech_reason = tech_result.get('reason', '暂无分析')
        
        result_text = f"""## 📊 双Agent综合分析 - {symbol}

---

### 🤖 Morefit (基本面分析)

**投票**: {mf_vote} | **建议仓位**: {mf_target_pct*100:.0f}% | **置信度**: {mf_confidence*100:.0f}%

**分析理由**:
{mf_reason}

---

### 📈 Technical_Nerd (技术面分析)

**投票**: {tech_vote_val} | **建议仓位**: {tech_target_pct*100:.0f}% | **置信度**: {tech_confidence*100:.0f}%

**分析理由**:
{tech_reason}

---

### 🎯 最终投票结果

**综合决策**: {final_vote} (置信度: {confidence}%)  
**建议仓位**: {target_position_pct*100:.0f}%  
**操作建议**: {suggestion}
"""
        
        return {
            "result": result_text,
            "final_decision": final_decision,
            "detailed_analysis": detailed_analysis,
            "all_tool_chains": all_tool_chains,
            "stock": symbol,
            "status": "completed"
        }
    
    def _clarify(self, state: GraphStatus) -> dict:
        """澄清节点 - 当需要用户提供更多信息时"""
        print(f"\n┌─ [clarify_node] 需要澄清")
        
        # 从 parse_result 中获取澄清问题
        clarification_question = state.get('parse_result', {}).get(
            'clarification', 
            "抱歉，我没有理解您的问题。请提供更多关于股票名称或分析需求的信息。"
        )
        
        count = state.get("clarification_count", 0) + 1
        print(f"│  澄清问题: {clarification_question}")
        print(f"│  澄清次数: {count}/3")
        print(f"└─ ⏸️ 中断: 请用户回复后，用 resume() 继续")
        
        # 返回状态并设置中断标志（不要抛异常，让 graph 正常结束但标记等待状态）
        return {
            "result": clarification_question,
            "status": "waiting_for_clarification",
            "clarification_count": count
        }

    async def run(self, user_input: str, thread_id: str = None) -> dict:
        """运行图 - 全新会话（异步版本）"""
        thread_id = thread_id or "default_thread"
        config = {"configurable": {"thread_id": thread_id}}
        
        # 检查是否是澄清后的重新运行
        current_state = await self.graph.aget_state(config)
        if current_state and current_state.values.get("status") == "waiting_for_clarification":
            # 说明之前在等待澄清，现在用户回复了
            print(f"\n{'='*60}")
            print(f"🔄 澄清回复后重新运行 | 输入: '{user_input}'")
            print(f"{'='*60}")
            # 清空之前的状态，重新运行（保留 clarification_count）
            count = current_state.values.get("clarification_count", 0)
        else:
            print(f"\n{'='*60}")
            print(f"🚀 开始运行 Graph | 输入: '{user_input}'")
            print(f"{'='*60}")
            count = 0
        
        # 全新状态运行
        initial_state = {
            "input": user_input,
            "parse_result": None,
            "result": None,
            "status": "init",
            "thread_id": thread_id,
            "next_agent": None,
            "clarification_count": count,
            "morefit_vote": None,
            "tech_vote": None,
            "final_decision": None
        }
        
        async for event in self.graph.astream(initial_state, config=config):
            pass
        
        # 检查是否需要澄清
        final = await self.graph.aget_state(config)
        final_values = dict(final.values) if final else {}
        
        if final_values.get("status") == "waiting_for_clarification":
            print(f"\n⏸️ 等待用户澄清...")
            return final_values
        
        print(f"\n{'='*60}")
        print(f"✅ Graph 运行完成")
        print(f"{'='*60}")
        
        return final_values

    async def resume(self, user_input: str, thread_id: str) -> dict:
        """恢复逻辑 - 用户回复澄清问题后重新运行（异步版本）"""
        print(f"\n🔄 [resume] 用户回复澄清 | thread: {thread_id}")
        print(f"   用户回复: '{user_input}'")
        # 直接把用户回复传给 run，run 会检测到 waiting 状态并处理
        return await self.run(user_input, thread_id)


# ==================== 测试脚本 ====================

def test_with_clarification_flow():
    """测试澄清流程 - 模拟用户需要澄清的场景"""
    print("\n" + "="*60)
    print("测试场景: 需要澄清的对话流程")
    print("="*60)
    print("\n💡 流程说明:")
    print("   1. 用户输入模糊问题 -> 系统要求澄清")
    print("   2. 用户补充信息 -> 系统继续处理")
    print("   3. 最终返回分析结果")
    
    checkpointer = MemorySaver()
    model = ChatTongyi(api_key=qianwen_api_key, temperature=0.3)
    preprocessor = Preprocessor(model, checkpointer)
    router = Router()
    
    # 创建 Agent 实例 - 使用统一股票工具（自动识别美股/A股）
    tech_agent = TECHNICAL_NERD(
        model=model,
        tools=[
            get_stock_price,          # 获取历史价格（技术面分析核心）
            get_stock_basic_info      # 获取基础信息
        ],
        checkpointer=checkpointer
    )
    morefit_agent = Morefit(
        model=model,
        tools=[
            get_stock_company_info,       # 获取公司详情（业务描述、CEO等）
            get_stock_financial_report_links,  # 获取财报链接
            get_stock_price,              # 获取价格（用于估值分析）
            get_stock_basic_info          # 获取基础信息
        ],
        checkpointer=checkpointer
    )
    
    fin_graph = FinGraph(
        preprocessor=preprocessor,
        router=router,
        agent={
            'TECHNICAL_NERD': tech_agent,
            'Morefit': morefit_agent
        },
        checkpointer=checkpointer
    )
    
    thread_id = "test_clarify_001"
    
    # ========== 第 1 轮：模糊输入触发澄清 ==========
    print("\n" + "-"*60)
    print("[第1轮] 用户输入: '分析一下这个股票'")
    print("-"*60)
    
    result1 = asyncio.run(fin_graph.run("分析一下这个股票", thread_id))
    
    if result1.get("status") == "waiting_for_clarification":
        # ========== 第 2 轮：用户澄清回复 ==========
        print("\n" + "-"*60)
        print("[第2轮] 用户看到系统提示后回复: '茅台'  ← 补充股票名称")
        print("-"*60)
        
        result2 = asyncio.run(fin_graph.resume("贵州茅台", thread_id))
        
        print("\n" + "-"*60)
        print("📋 最终结果:")
        print("-"*60)
        print(f"   状态: {result2.get('status')}")
        print(f"   结果: {result2.get('result')}")
    else:
        print("\n⚠️ 注意: 没有触发澄清，直接完成了")
        print(f"   状态: {result1.get('status')}")
        print(f"   结果: {result1.get('result')}")


def test_normal_flow():
    """测试正常流程"""
    print("\n" + "="*60)
    print("测试场景: 正常直接完成的流程")
    print("="*60)
    
    checkpointer = MemorySaver()
    model = ChatTongyi(api_key=qianwen_api_key, temperature=0.3)
    preprocessor = Preprocessor(model, checkpointer)
    router = Router()
    
    # 创建 Agent 实例 - 使用统一股票工具（自动识别美股/A股）
    tech_agent = TECHNICAL_NERD(
        model=model,
        tools=[
            get_stock_price,          # 获取历史价格（技术面分析核心）
            get_stock_basic_info      # 获取基础信息
        ],
        checkpointer=checkpointer
    )
    morefit_agent = Morefit(
        model=model,
        tools=[
            get_stock_company_info,       # 获取公司详情（业务描述、CEO等）
            get_stock_financial_report_links,  # 获取财报链接
            get_stock_price,              # 获取价格（用于估值分析）
            get_stock_basic_info          # 获取基础信息
        ],
        checkpointer=checkpointer
    )
    
    fin_graph = FinGraph(
        preprocessor=preprocessor,
        router=router,
        agent={
            'TECHNICAL_NERD': tech_agent,
            'Morefit': morefit_agent
        },
        checkpointer=checkpointer
    )
    
    # 打印图结构
    print("\n📊 Graph 结构 (Mermaid):")
    print(fin_graph.graph.get_graph().draw_mermaid())
    
    test_cases = [
        ("分析一下茅台的K线走势", "thread_normal_1"),      # 技术面 -> TECHNICAL_NERD
        ("寒武纪股票估值高吗？建议买入吗？", "thread_normal_3"),  # 基本面/建议 -> Morefit
    ]
    
    for user_input, thread_id in test_cases:
        print("\n" + "="*60)
        result = asyncio.run(fin_graph.run(user_input, thread_id))
        print(f"\n📋 最终结果:")
        print(f"   status: {result.get('status')}")
        print(f"   result: {result.get('result')[:50]}..." if result.get('result') and len(result.get('result')) > 50 else f"   result: {result.get('result')}")


if __name__ == "__main__":
    print("="*60)
    print("FinGraph 测试脚本（使用真实 Preprocessor + Router）")
    print("="*60)
    
    # 选择测试场景
    test_normal_flow()              # 正常流程
    test_with_clarification_flow()  # 澄清流程
    
    print("\n" + "="*60)
    print("测试完成!")
    print("="*60)
