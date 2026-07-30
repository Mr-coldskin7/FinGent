from datetime import datetime
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage
from dotenv import load_dotenv
from langgraph.checkpoint.mysql.pymysql import PyMySQLSaver
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Iterator
import os
import uuid
import asyncio

try:
    import LLM.tools as tools
except ImportError:
    import tools

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
        system_prompt: str,
        checkpointer=None,
        simulated_date: Optional[str] = None,
        memory_manager=None,
        user_id: Optional[str] = None,
    ):
        self.checkpointer = checkpointer
        self.thread_id = str(uuid.uuid4())
        self.model = model
        self.tools = tools
        self.memory_manager = memory_manager
        self.user_id = user_id or "anonymous"

        # 支持回测时的日期模拟（环境变量优先级最高，用于回测）
        effective_date = (
            os.getenv("FINGENT_SIMULATED_DATE") or simulated_date or current_date
        )
        self.reminder = f"""### ⚠️ TODAY IS {effective_date} (YYYY-MM-DD).
NEVER GUESS DATES. You are also a professional financial assistant."""

        self.agent = create_agent(
            model=model,
            tools=tools,
            system_prompt=self.reminder + system_prompt,
            checkpointer=self.checkpointer,
        )

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

        for attempt in range(max_retries):
            try:
                response = self.agent.invoke({"messages": [new_message]}, config=config)
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

        for attempt in range(max_retries):
            try:
                response = await self.agent.ainvoke(
                    {"messages": [new_message]}, config=config
                )
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


# ==================== Usage Examples ====================

if __name__ == "__main__":
    # 初始化
    _mysql_saver_ctx = PyMySQLSaver.from_conn_string(
        os.getenv("MYSQL_URL", "mysql+pymysql://root:password@localhost:3306/fingent")
    )
    checkpointer = _mysql_saver_ctx.__enter__()
    checkpointer.setup()
    model = ChatOpenAI(
        api_key=qianwen_api_key,
        base_url=os.getenv(
            "MODEL_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
        ),
        model="qwen-max",
        temperature=0.3,
    )

    agent = Agent(
        model=model,
        tools=[tools.get_us_stock_price],
        system_prompt="You are a helpful assistant.",
        checkpointer=checkpointer,
    )

    # 示例 1：单次对话
    resp = agent.chat("查一下英伟达股价")
    print(f"Answer: {resp.final_answer}")
    print(f"Steps: {resp.step_count}")

    # 示例 3：获取历史状态（即使在另一个函数/进程中）
    current_state = agent.get_state()
    print(f"Current thread has {current_state.step_count} messages")

    # 示例 4：切换回之前的线程继续对话
    previous_thread_id = resp.thread_id  # 使用之前的线程ID
    agent.switch_thread(previous_thread_id)
    print(f"Switched to: {previous_thread_id}")

    history = agent.get_state()
    if history:
        print(history.show_chain())  # 会显示英伟达对话历史
    resp = agent.chat("再查一下苹果公司的股价")
    thread_id = resp.thread_id
    print(f"Answer: {resp.final_answer}")
    print(f"Steps: {resp.step_count}")
    print(f"thread ID: {thread_id}")
    print(resp.show_chain())
