from datetime import datetime
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import os
import uuid
import asyncio

# deepagents（create_deep_agent + skills），缺失时回退到 create_agent
try:
    from deepagents import create_deep_agent
    from langchain.agents.middleware.types import AgentMiddleware
except ImportError:
    create_deep_agent = None
    AgentMiddleware = None

# deepagents 内置 FilesystemMiddleware/TodoListMiddleware/SubAgentMiddleware 会附带
# 一批文件系统/待办/子代理工具，对纯金融 agent 是多余工具面（也会干扰模型选工具）。
# create_deep_agent 无关闭开关（middleware 参数只能追加），故用公开扩展点
# AgentMiddleware.wrap_model_call + ModelRequest.override 在模型请求层过滤。
# 必须保留 read_file：技能是渐进式披露，agent 靠 read_file 读 SKILL.md。
_UNWANTED_DEEPAGENT_TOOLS = frozenset(
    {"ls", "write_file", "edit_file", "glob", "grep", "execute", "write_todos", "task"}
)


def _trim_tool_surface(request):
    """剔除 deepagents 附带的多余工具，只保留 read_file 与业务工具。"""
    tools = getattr(request, "tools", None)
    if not tools:
        return request
    kept = [
        t for t in tools if getattr(t, "name", None) not in _UNWANTED_DEEPAGENT_TOOLS
    ]
    return request if len(kept) == len(tools) else request.override(tools=kept)


if AgentMiddleware is not None:

    class _ToolSurfaceMiddleware(AgentMiddleware):
        """只保留 read_file/业务工具，去掉 deepagents 的 8 个多余工具。"""

        def wrap_model_call(self, request, handler):  # sync 路径
            return handler(_trim_tool_surface(request))

        async def awrap_model_call(self, request, handler):  # async 路径（图实际走这里）
            return await handler(_trim_tool_surface(request))

# 技能目录：LLM/skills/<skill_name>/SKILL.md
SKILLS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "skills")
SKILLS_SOURCE = "/skills/fingent/"

try:
    from Data.memory import get_memory_manager, MemoryManager
except ImportError:
    get_memory_manager = None
    MemoryManager = None

load_dotenv()
qianwen_api_key = os.getenv("QIANWEN_API_KEY")
datetime_format = "%Y-%m-%d"
current_date = datetime.now().strftime(datetime_format)


@dataclass
class AgentResponse:
    """Standard response format for Agent conversations"""

    final_answer: str
    messages: List[BaseMessage]
    thread_id: str
    step_count: int
    tool_calls: int

    def show_chain(self) -> str:
        """Return formatted CoT string (for printing or logging)"""
        lines = [
            f"\n=== Thread: {self.thread_id} (Steps: {self.step_count}, Tools: {self.tool_calls}) ==="
        ]

        for i, msg in enumerate(self.messages, 1):
            if isinstance(msg, HumanMessage):
                lines.append(f"{i}. 👤 User: {msg.content}")
            elif isinstance(msg, AIMessage):
                if msg.tool_calls:
                    lines.append(
                        f"{i}. 💭 Thought: {msg.content[:80]}..."
                        if msg.content
                        else f"{i}. 💭 Thought: [Decided to use tools]"
                    )
                    for tool in msg.tool_calls:
                        args = tool.get("args", {})
                        lines.append(f"   └─ 🛠️  Action: {tool.get('name')}({args})")
                else:
                    lines.append(f"{i}. ✅ Final: {msg.content}")
            elif isinstance(msg, ToolMessage):
                content = str(msg.content)
                if len(content) > 100:
                    content = content[:100] + "..."
                lines.append(f"{i}. 👁️  Observation: {content}")

        lines.append("=" * 50)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format (for serialization)"""
        return {
            "thread_id": self.thread_id,
            "final_answer": self.final_answer,
            "step_count": self.step_count,
            "tool_calls": self.tool_calls,
            "messages": [
                {
                    "type": msg.type,
                    "content": msg.content if hasattr(msg, "content") else str(msg),
                    "tool_calls": getattr(msg, "tool_calls", None),
                }
                for msg in self.messages
            ],
        }


class Agent:
    """
    Base Agent class with Checkpointer support
    Features:
    - No local chat_history cache, fully relies on checkpointer as single source of truth
    - All methods return standardized AgentResponse format
    - Supports multi-thread conversations (via thread_id)
    - Provides complete Chain-of-Thought (CoT) tracking
    - Includes persistent state management
    - Offers flexible thread switching and management
    """

    def __init__(
        self,
        model,
        tools,
        system_prompt: str = "",
        checkpointer=None,
        simulated_date: Optional[str] = None,
        memory_manager=None,
        user_id: Optional[str] = None,
        skill_name: Optional[str] = None,
    ):
        self.checkpointer = checkpointer
        self.thread_id = str(uuid.uuid4())
        self.model = model
        self.tools = tools
        self.memory_manager = memory_manager
        self.user_id = user_id or "anonymous"
        self.skill_name = skill_name

        # 支持回测时的日期模拟（环境变量优先级最高，用于回测）
        effective_date = (
            os.getenv("FINGENT_SIMULATED_DATE") or simulated_date or current_date
        )
        self.reminder = f"""### ⚠️ TODAY IS {effective_date} (YYYY-MM-DD).
NEVER GUESS DATES. You are also a professional financial assistant."""

        if skill_name and create_deep_agent is not None:
            # ═══ deepagents 模式：身份/协议/输出格式在技能文件里，prompt 与 tools 分离 ═══
            self._skill_files = self._load_skill(skill_name)
            skill_path = f"{SKILLS_SOURCE}{skill_name}/SKILL.md"
            skill_system_prompt = (
                self.reminder
                + f"\n\nYou are the **{skill_name}** agent of FinGent.\n"
                f"Your full identity, analysis protocol, and required JSON output "
                f"schema live in your skill file:\n\n`{skill_path}`\n\n"
                f"### MANDATORY\nRead your skill file **first** (use the `read_file` "
                f"tool with that path) before answering. Your final answer MUST be "
                f"valid JSON matching the schema defined in the skill."
            )
            self.agent = create_deep_agent(
                model=model,
                tools=tools,
                system_prompt=skill_system_prompt,
                skills=[SKILLS_SOURCE],
                checkpointer=self.checkpointer,
                middleware=[_ToolSurfaceMiddleware()] if AgentMiddleware is not None else [],
            )
        else:
            # ═══ 传统模式：system_prompt 直接注入（无技能文件时回退）═══════
            self._skill_files = {}
            self.agent = create_agent(
                model=model,
                tools=tools,
                system_prompt=self.reminder + (system_prompt or ""),
                checkpointer=self.checkpointer,
            )

    def _load_skill(self, skill_name: str) -> Dict[str, Dict[str, Any]]:
        """读取 LLM/skills/<skill_name>/SKILL.md → StateBackend files 字典。

        deepagents 默认 StateBackend 下，技能文件通过 invoke(files=...) 注入到
        进程内虚拟文件系统（不碰真实磁盘文件）。agent 用 read_file 读取。
        """
        skill_path = os.path.join(SKILLS_DIR, skill_name, "SKILL.md")
        if not os.path.exists(skill_path):
            print(f"⚠️ Skill file not found: {skill_path}")
            return {}
        with open(skill_path, encoding="utf-8") as f:
            content = f.read()
        # StateBackend 的 files 状态要求 FileData 结构（content 为行列表 + 时间戳）
        from deepagents.backends.utils import create_file_data

        return {f"{SKILLS_SOURCE}{skill_name}/SKILL.md": create_file_data(content)}

    def _get_config(self, thread_id: Optional[str] = None) -> dict:
        tid = thread_id or self.thread_id
        return {"configurable": {"thread_id": tid}}

    def _build_response(
        self, messages: List[BaseMessage], thread_id: str
    ) -> AgentResponse:
        """Build AgentResponse from message list"""
        final_msg = messages[-1] if messages else None
        final_content = (
            final_msg.content if hasattr(final_msg, "content") else str(final_msg)
        )

        tool_call_count = sum(
            1 for m in messages if isinstance(m, AIMessage) and m.tool_calls
        )

        return AgentResponse(
            final_answer=final_content,
            messages=messages,
            thread_id=thread_id,
            step_count=len(messages),
            tool_calls=tool_call_count,
        )

    def format_prompt(self, stock: str, user_input: str) -> str:
        """
        Agent 可覆盖此方法自定义 prompt 格式。
        Graph 层调度时统一调用，避免在 graph 里写死 prompt 模板。
        """
        return f"请分析股票：{stock}\n\n用户问题：{user_input}"

    def _inject_memory_sync(
        self, user_input: str, stock: Optional[str] = None, user_id: Optional[str] = None
    ) -> str:
        """Synchronous memory injection (fallback when aiosqlite unavailable)."""
        if not self.memory_manager or not stock:
            return user_input
        try:
            uid = user_id or self.user_id or "anonymous"
            agent_name = self.__class__.__name__
            # sync fallback: use sync methods on MemoryManager
            if hasattr(self.memory_manager, "build_memory_context"):
                import asyncio
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = None
                if loop:
                    # Cannot run async in running loop; skip injection
                    return user_input
                memory_ctx = asyncio.run(
                    self.memory_manager.build_memory_context(uid, stock, agent_name)
                )
                if memory_ctx:
                    return f"【相关记忆背景】\n{memory_ctx}\n\n---\n\n{user_input}"
        except Exception:
            pass
        return user_input

    async def _inject_memory_async(
        self, user_input: str, stock: Optional[str] = None, user_id: Optional[str] = None
    ) -> str:
        """Asynchronous memory injection."""
        if not self.memory_manager or not stock:
            return user_input
        try:
            uid = user_id or self.user_id or "anonymous"
            agent_name = self.__class__.__name__
            memory_ctx = await self.memory_manager.build_memory_context(
                uid, stock, agent_name
            )
            if memory_ctx:
                return f"【相关记忆背景】\n{memory_ctx}\n\n---\n\n{user_input}"
        except Exception as e:
            print(f"⚠️ Memory injection failed: {e}")
        return user_input

    def chat(
        self,
        user_input: str,
        thread_id: Optional[str] = None,
        stock: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> AgentResponse:
        """
        Execute conversation, automatically continues based on history stored in checkpointer
        Returns standardized AgentResponse with complete Chain-of-Thought tracking

        Args:
            user_input: Input message from user
            thread_id: Optional thread ID to switch to before chatting
            stock: Stock symbol for memory injection
            user_id: User identifier for memory injection

        Returns:
            AgentResponse: Standardized response with CoT tracking
        """
        import time

        if thread_id:
            self.thread_id = thread_id

        config = self._get_config()

        # Inject L3 memory if available (sync fallback)
        enriched_input = self._inject_memory_sync(user_input, stock, user_id)

        # Pass only new message, LangGraph automatically loads history from checkpointer
        new_message = HumanMessage(content=enriched_input)

        # 重试机制：最多3次，带指数退避延迟
        max_retries = 3
        base_delay = 1.0  # 基础延迟1秒

        invoke_input: Dict[str, Any] = {"messages": [new_message]}
        if self._skill_files:
            invoke_input["files"] = self._skill_files  # deepagents skills (StateBackend)

        for attempt in range(max_retries):
            try:
                response = self.agent.invoke(invoke_input, config=config)
                return self._build_response(response["messages"], self.thread_id)
            except Exception as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2**attempt)  # 指数退避: 1, 2, 4秒
                    print(
                        f"⚠️ API调用失败 (尝试 {attempt + 1}/{max_retries}): {str(e)[:50]}"
                    )
                    print(f"   等待 {delay:.1f} 秒后重试...")
                    time.sleep(delay)
                else:
                    print(f"❌ API调用最终失败: {str(e)[:100]}")
                    raise

    async def achat(
        self,
        user_input: str,
        thread_id: Optional[str] = None,
        stock: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> AgentResponse:
        """
        Execute conversation, automatically continues based on history stored in checkpointerasynchronously
        Returns standardized AgentResponse with complete Chain-of-Thought tracking

        Args:
            user_input: Input message from user
            thread_id: Optional thread ID to switch to before chatting
            stock: Stock symbol for memory injection
            user_id: User identifier for memory injection

        Returns:
            AgentResponse: Standardized response with CoT tracking
        """
        if thread_id:
            self.thread_id = thread_id

        config = self._get_config()

        # Inject L3 memory if available
        enriched_input = await self._inject_memory_async(user_input, stock, user_id)

        # Pass only new message, LangGraph automatically loads history from checkpointer
        new_message = HumanMessage(content=enriched_input)

        # 重试机制：最多3次，带指数退避延迟
        max_retries = 3
        base_delay = 1.0  # 基础延迟1秒

        invoke_input: Dict[str, Any] = {"messages": [new_message]}
        if self._skill_files:
            invoke_input["files"] = self._skill_files  # deepagents skills (StateBackend)

        for attempt in range(max_retries):
            try:
                response = await self.agent.ainvoke(invoke_input, config=config)
                return self._build_response(response["messages"], self.thread_id)
            except Exception as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2**attempt)  # 指数退避: 1, 2, 4秒
                    print(
                        f"⚠️ API调用失败 (尝试 {attempt + 1}/{max_retries}): {str(e)[:50]}"
                    )
                    print(f"   等待 {delay:.1f} 秒后重试...")
                    await asyncio.sleep(delay)
                else:
                    print(f"❌ API调用最终失败: {str(e)[:100]}")
                    raise

    def get_state(self, thread_id: Optional[str] = None) -> Optional[AgentResponse]:
        """
        Get current full state of specified thread (from checkpointer)
        Returns None if thread doesn't exist

        Args:
            thread_id: Thread ID to retrieve state for (uses current if None)

        Returns:
            AgentResponse: Complete conversation state with CoT tracking, or None
        """
        config = self._get_config(thread_id)
        state = self.checkpointer.get(config)

        if not state or "channel_values" not in state:
            return None

        messages = state["channel_values"].get("messages", [])
        if not messages:
            return None

        return self._build_response(messages, thread_id or self.thread_id)

    def get_history_summary(self, thread_id: Optional[str] = None) -> Dict[str, Any]:
        """Get conversation history summary (for quick overview)"""
        resp = self.get_state(thread_id)
        if not resp:
            return {"error": "No history found"}

        return {
            "thread_id": resp.thread_id,
            "total_messages": resp.step_count,
            "tool_calls": resp.tool_calls,
            "user_turns": len(
                [m for m in resp.messages if isinstance(m, HumanMessage)]
            ),
            "last_message": (
                resp.final_answer[:100] + "..."
                if len(resp.final_answer) > 100
                else resp.final_answer
            ),
        }

    def switch_thread(self, thread_id: str) -> AgentResponse:
        """
        Switch to specified thread, return current state of that thread
        If thread doesn't exist, return empty state but switch thread_id

        Args:
            thread_id: Target thread ID to switch to

        Returns:
            AgentResponse: Current state of the thread (existing or empty for new)
        """
        self.thread_id = thread_id
        state = self.get_state(thread_id)

        if state:
            print(
                f"Switched to existing thread: {thread_id} ({state.step_count} messages)"
            )
            return state
        else:
            print(f"New thread created: {thread_id}")
            # Return empty structure for new session
            return AgentResponse(
                final_answer="",
                messages=[],
                thread_id=thread_id,
                step_count=0,
                tool_calls=0,
            )

    def reset_thread(self, new_thread_id: Optional[str] = None) -> str:
        """
        Reset current thread (generates new ID, old data remains in checkpointer but becomes inaccessible)
        Returns new thread_id

        Args:
            new_thread_id: Optional new thread ID (generates UUID if None)

        Returns:
            str: New thread ID
        """
        old_id = self.thread_id
        self.thread_id = new_thread_id or str(uuid.uuid4())
        print(f"Thread reset: {old_id} → {self.thread_id}")
        return self.thread_id

    def continously_chat(self):
        """Continuous chat mode with AgentResponse handling display"""
        print(f"Current Thread ID: {self.thread_id}")
        print("Commands:")
        print("  /history    - Show full CoT chain")
        print("  /summary    - Show session summary")
        print("  /switch <id>- Switch to thread")
        print("  /reset      - Start new thread")
        print("  exit/quit/q - Exit")
        print("-" * 50)

        while True:
            user_input = input("\nYou: ").strip()

            if not user_input:
                continue

            # Command processing
            if user_input.lower() in ["exit", "quit", "q"]:
                break
            elif user_input == "/history":
                state = self.get_state()
                if state:
                    print(state.show_chain())
                else:
                    print("No history in current thread")
                continue
            elif user_input == "/summary":
                summary = self.get_history_summary()
                print(f"\nSession Summary: {summary}")
                continue
            elif user_input.startswith("/switch "):
                tid = user_input.split(" ", 1)[1]
                self.switch_thread(tid)
                continue
            elif user_input == "/reset":
                self.reset_thread()
                continue

            # Normal conversation
            try:
                resp = self.chat(user_input)
                print(f"\nAI: {resp.final_answer}")

                # Optional: show brief CoT for this interaction (if tools were used)
                if resp.tool_calls > 0:
                    print(
                        f"\n[Used {resp.tool_calls} tool(s), {resp.step_count} steps total]"
                    )

            except Exception as e:
                print(f"Error: {e}")

    def list_all_messages(self, thread_id: Optional[str] = None) -> List[BaseMessage]:
        """
        Raw interface: Directly get message list (for advanced operations)

        Args:
            thread_id: Thread ID to retrieve messages for (uses current if None)

        Returns:
            List[BaseMessage]: List of all messages in the thread
        """
        config = self._get_config(thread_id)
        state = self.checkpointer.get(config)
        if state and "channel_values" in state:
            return state["channel_values"].get("messages", [])
        return []
