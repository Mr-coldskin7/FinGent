from langgraph.checkpoint.mysql.pymysql import PyMySQLSaver
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages import HumanMessage, AIMessage

def chatbot(state: MessagesState):
    """简单bot，回复最后一条消息"""
    last_message = state["messages"][-1].content
    return {"messages": [AIMessage(content=f"收到: {last_message}")]}

# MySQL连接（根据你的环境修改）
DB_URI = "mysql+pymysql://root:123456@localhost:3333/mysql"
# 如果用Docker 3307端口：DB_URI = "mysql+pymysql://root:123456@localhost:3307/mysql"

print("=== MySQL持久化测试 ===")

with PyMySQLSaver.from_conn_string(DB_URI) as checkpointer:
    # 第一次必须调用，创建表结构
    print("1. 初始化数据库...")
    checkpointer.setup()
    print("   ✓ 表创建成功")
    
    # 建图
    workflow = StateGraph(MessagesState)
    workflow.add_node("chatbot", chatbot)
    workflow.add_edge(START, "chatbot")
    workflow.add_edge("chatbot", END)
    
    app = workflow.compile(checkpointer=checkpointer)
    
    # 第一轮对话
    print("\n2. 第一轮对话...")
    config = {"configurable": {"thread_id": "user_mysql_001", "checkpoint_ns": ""}}
    result = app.invoke(
        {"messages": [HumanMessage(content="MySQL测试")]}, 
        config=config
    )
    print(f"   Bot: {result['messages'][-1].content}")
    
    # 第二轮对话（验证持久化恢复）
    print("\n3. 第二轮对话（验证恢复）...")
    result2 = app.invoke(
        {"messages": [HumanMessage(content="还记得我吗")]}, 
        config=config
    )
    print(f"   历史消息数: {len(result2['messages'])}")
    for i, msg in enumerate(result2['messages'], 1):
        sender = "User" if isinstance(msg, HumanMessage) else "Bot"
        print(f"   {i}. {sender}: {msg.content}")
    
    # 查看数据库中的checkpoints
    print("\n4. 查看数据库记录...")
    all_checkpoints = list(checkpointer.list(config))
    print(f"   该用户共有 {len(all_checkpoints)} 个checkpoint")

print("\n=== MySQL持久化验证完成 ===")