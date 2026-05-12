from langgraph.checkpoint.redis import RedisSaver
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages import HumanMessage, AIMessage
import redis


def chatbot(state: MessagesState):
    last_message = state["messages"][-1].content
    return {"messages": [AIMessage(content=f"收到: {last_message}")]}


# 关键：先清空可能存在的旧数据，让LangGraph重新初始化索引
r = redis.Redis(host="localhost", port=6379)
r.flushall()  # 清空Redis
print("✅ Redis已清空")

# 注意：需要手动调用 setup() 来重新创建索引，或者让 LangGraph 自动创建
# 方式1：使用 setup 显式初始化
with RedisSaver.from_conn_string("redis://localhost:6379") as checkpointer:
    # 显式调用 setup 创建索引（如果 flushall 后需要）
    checkpointer.setup()
    print("✅ Redis索引已初始化")

    workflow = StateGraph(MessagesState)
    workflow.add_node("chatbot", chatbot)
    workflow.add_edge(START, "chatbot")
    workflow.add_edge("chatbot", END)

    app = workflow.compile(checkpointer=checkpointer)

    # 第一轮
    print("\n=== 第一轮 ===")
    config = {"configurable": {"thread_id": "user_redis_001"}}
    result = app.invoke(
        {"messages": [HumanMessage(content="Redis测试")]}, config=config
    )
    print(f"Bot: {result['messages'][-1].content}")

    # 第二轮 - 测试状态恢复
    print("\n=== 第二轮（自动恢复） ===")
    result2 = app.invoke(
        {"messages": [HumanMessage(content="还记得我吗")]}, config=config
    )
    print(f"历史消息数: {len(result2['messages'])}")
    for i, msg in enumerate(result2["messages"], 1):
        sender = "User" if isinstance(msg, HumanMessage) else "Bot"
        print(f"{i}. {sender}: {msg.content}")

print("\n=== Redis持久化验证完成 ===")
