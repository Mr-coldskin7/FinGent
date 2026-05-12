"""
FinGraph 可视化工具
支持 Mermaid 图、Graphviz 流程图、网络拓扑图
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# 方法1: LangGraph 自带 Mermaid 图（最简单，推荐）
# ============================================================


def visualize_mermaid():
    """生成 Mermaid 格式的流程图"""
    from langgraph.checkpoint.mysql.pymysql import PyMySQLSaver
    from langchain_openai import ChatOpenAI
    from LLM.preprocess import Preprocessor
    from LLM.router import Router
    from LLM.agent import TECHNICAL_NERD, Morefit
    from LLM.graph import FinGraph
    from LLM.unified_stock_tools import (
        get_stock_price,
        get_stock_basic_info,
        get_stock_company_info,
        get_stock_financial_report_links,
        get_stock_financial_statements,
    )

    # 初始化组件
    _mysql_saver_ctx = PyMySQLSaver.from_conn_string(
        os.getenv("MYSQL_URL", "mysql+pymysql://root:password@localhost:3306/fingent")
    )
    checkpointer = _mysql_saver_ctx.__enter__()
    checkpointer.setup()
    model = ChatOpenAI(
        api_key=os.getenv("QIANWEN_API_KEY"),
        base_url=os.getenv(
            "MODEL_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
        ),
        model="qwen-max",
        temperature=0.3,
    )

    preprocessor = Preprocessor(model, checkpointer)
    router = Router()

    tech_agent = TECHNICAL_NERD(
        model=model,
        tools=[get_stock_price, get_stock_basic_info],
        checkpointer=checkpointer,
    )

    morefit_agent = Morefit(
        model=model,
        tools=[
            get_stock_company_info,
            get_stock_financial_report_links,
            get_stock_financial_statements,
            get_stock_price,
            get_stock_basic_info,
        ],
        checkpointer=checkpointer,
    )

    fin_graph = FinGraph(
        preprocessor=preprocessor,
        router=router,
        agent={"TECHNICAL_NERD": tech_agent, "Morefit": morefit_agent},
        checkpointer=checkpointer,
    )

    # 生成 Mermaid 图
    mermaid_code = fin_graph.graph.get_graph().draw_mermaid()

    print("=" * 60)
    print("FinGraph Mermaid 流程图")
    print("=" * 60)
    print(mermaid_code)
    print("\n" + "=" * 60)
    print("提示：将上述代码粘贴到 https://mermaid.live/ 查看图形")
    print("=" * 60)

    return mermaid_code


# ============================================================
# 方法2: Graphviz 可视化（更美观，可导出图片）
# ============================================================


def visualize_graphviz():
    """使用 Graphviz 绘制流程图（需要先安装 graphviz: pip install graphviz）"""
    try:
        from graphviz import Digraph
    except ImportError:
        print("请先安装 graphviz: pip install graphviz")
        print("同时需要安装系统 graphviz: https://graphviz.org/download/")
        return None

    dot = Digraph(comment="FinGraph Architecture", format="png")
    dot.attr(rankdir="TB", size="12,10", dpi="150")

    # 设置节点样式
    dot.attr("node", shape="box", style="rounded,filled", fontname="Microsoft YaHei")

    # 开始节点
    dot.node("start", "START", fillcolor="#90EE90", shape="ellipse")

    # 预处理节点
    dot.node("preprocess", "Preprocessor\n(输入解析)", fillcolor="#87CEEB")

    # 路由节点
    dot.node("route", "Router\n(意图路由)", fillcolor="#FFD700")

    # 两个 Agent 节点
    dot.node(
        "tech",
        "TECHNICAL_NERD\n(技术面分析)\n基于价格和指标",
        fillcolor="#DDA0DD",
        shape="box3d",
    )
    dot.node(
        "fund",
        "Morefit\n(基本面分析)\n基于财报和估值",
        fillcolor="#98FB98",
        shape="box3d",
    )

    # 双 Agent 模式
    dot.node("all", "ALL Mode\n(双Agent并行)", fillcolor="#F0A0A0")
    dot.node("voting", "Voting\n(投票汇总)", fillcolor="#FFA500")

    # 澄清节点
    dot.node("clarify", "Clarify\n(澄清节点)", fillcolor="#FFB6C1")

    # 结束节点
    dot.node("end", "END", fillcolor="#FF6B6B", shape="ellipse")

    # 添加边
    dot.edge("start", "preprocess")
    dot.edge("preprocess", "route")

    # 路由分支
    dot.edge("route", "tech", label="技术面请求")
    dot.edge("route", "fund", label="基本面请求")
    dot.edge("route", "all", label="回测/综合")
    dot.edge("route", "clarify", label="需要澄清", style="dashed")

    # 单 Agent 直接结束
    dot.edge("tech", "end")
    dot.edge("fund", "end")

    # 双 Agent 流程
    dot.edge("all", "voting")
    dot.edge("voting", "end")

    # 澄清后重新路由（简化表示）
    dot.edge("clarify", "end", style="dashed", label="等待用户")

    # 保存并渲染
    output_path = "fingraph_architecture"
    dot.render(output_path, cleanup=True)
    print(f"Graphviz 图已保存: {output_path}.png")

    return dot


# ============================================================
# 方法3: Matplotlib + NetworkX（网络拓扑图）
# ============================================================


def visualize_networkx():
    """使用 NetworkX 和 Matplotlib 绘制网络图"""
    try:
        import matplotlib.pyplot as plt
        import networkx as nx
        import matplotlib
    except ImportError:
        print("请先安装: pip install matplotlib networkx")
        return None

    # 设置中文字体（解决乱码问题）
    plt.rcParams["font.sans-serif"] = [
        "SimHei",
        "Microsoft YaHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

    # 创建有向图
    G = nx.DiGraph()

    # 添加节点
    nodes = {
        "START": {"pos": (0, 4), "color": "#90EE90"},
        "Preprocessor": {"pos": (0, 3), "color": "#87CEEB"},
        "Router": {"pos": (0, 2), "color": "#FFD700"},
        "TECHNICAL\n_NERD": {"pos": (-2, 1), "color": "#DDA0DD"},
        "Morefit": {"pos": (2, 1), "color": "#98FB98"},
        "ALL Mode": {"pos": (0, 1), "color": "#F0A0A0"},
        "Voting": {"pos": (0, 0), "color": "#FFA500"},
        "Clarify": {"pos": (3, 2), "color": "#FFB6C1"},
        "END": {"pos": (0, -1), "color": "#FF6B6B"},
    }

    for node, attrs in nodes.items():
        G.add_node(node, **attrs)

    # 添加边
    edges = [
        ("START", "Preprocessor"),
        ("Preprocessor", "Router"),
        ("Router", "TECHNICAL\n_NERD"),
        ("Router", "Morefit"),
        ("Router", "ALL Mode"),
        ("Router", "Clarify"),
        ("TECHNICAL\n_NERD", "END"),
        ("Morefit", "END"),
        ("ALL Mode", "Voting"),
        ("Voting", "END"),
    ]
    G.add_edges_from(edges)

    # 绘制
    fig, ax = plt.subplots(figsize=(12, 8))

    # 获取位置
    pos = {node: attrs["pos"] for node, attrs in nodes.items()}
    colors = [attrs["color"] for node, attrs in nodes.items()]

    # 绘制节点
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=4000, alpha=0.9, ax=ax)

    # 绘制边
    nx.draw_networkx_edges(
        G,
        pos,
        edge_color="gray",
        arrows=True,
        arrowsize=20,
        arrowstyle="->",
        width=2,
        ax=ax,
    )

    # 绘制标签
    nx.draw_networkx_labels(G, pos, font_size=10, font_family="sans-serif", ax=ax)

    # 添加标题
    ax.set_title("FinGraph 系统架构图", fontsize=16, fontweight="bold", pad=20)
    ax.axis("off")

    # 添加图例
    legend_elements = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#90EE90",
            markersize=12,
            label="开始/结束",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#87CEEB",
            markersize=12,
            label="预处理",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#FFD700",
            markersize=12,
            label="路由",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#DDA0DD",
            markersize=12,
            label="技术面Agent",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#98FB98",
            markersize=12,
            label="基本面Agent",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#FFA500",
            markersize=12,
            label="投票决策",
        ),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.savefig(
        "fingraph_networkx.png",
        dpi=150,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    print("网络图已保存: fingraph_networkx.png")
    plt.show()

    return G


# ============================================================
# 方法4: ASCII 艺术图（命令行快速预览）
# ============================================================


def visualize_ascii():
    """ASCII 艺术图，适合命令行快速查看"""
    ascii_art = """
    ┌─────────────────────────────────────────────────────────────────┐
    │                     FinGraph 系统架构                            │
    └─────────────────────────────────────────────────────────────────┘
    
                            ┌─────────┐
                            │  START  │
                            └────┬────┘
                                 │
                                 ▼
                       ┌───────────────────┐
                       │   Preprocessor    │
                       │    (输入解析)      │
                       └─────────┬─────────┘
                                 │
                                 ▼
                         ┌───────────────┐
                         │    Router     │
                         │   (意图路由)   │
                         └───────┬───────┘
                                 │
            ┌────────────────────┼────────────────────┐
            │                    │                    │
            ▼                    ▼                    ▼
    ┌───────────────┐   ┌───────────────┐   ┌───────────────┐
    │TECHNICAL_NERD │   │    ALL Mode   │   │    Morefit    │
    │  (技术面分析)  │   │  (双Agent并行) │   │  (基本面分析)  │
    │               │   │               │   │               │
    │ • 价格数据    │   │               │   │ • 财务报表    │
    │ • 技术指标    │   │               │   │ • 公司信息    │
    └───────┬───────┘   └───────┬───────┘   └───────┬───────┘
            │                    │                    │
            │                    ▼                    │
            │           ┌───────────────┐             │
            │           │    Voting     │             │
            │           │   (投票汇总)   │             │
            │           └───────┬───────┘             │
            │                    │                    │
            └────────────────────┼────────────────────┘
                                 │
                                 ▼
                            ┌─────────┐
                            │   END   │
                            └─────────┘
    
    ═════════════════════════════════════════════════════════════════
    路由说明:
    • 技术面请求  → TECHNICAL_NERD
    • 基本面请求  → Morefit  
    • 回测/综合   → ALL Mode → Voting
    • 信息不全    → Clarify (澄清)
    ═════════════════════════════════════════════════════════════════
    """
    print(ascii_art)
    return ascii_art


# ============================================================
# 主函数
# ============================================================

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("FinGraph 可视化工具")
    print("=" * 60)
    print("\n可选方法:")
    print("1. mermaid  - 生成 Mermaid 流程图代码")
    print("2. graphviz - 生成 Graphviz 流程图 (需安装 graphviz)")
    print("3. networkx - 生成网络拓扑图 (需安装 matplotlib)")
    print("4. ascii    - ASCII 艺术图 (命令行预览)")
    print("5. all      - 运行所有可用的方法")
    print("\n用法: python visualize_fingraph.py [方法名]")
    print("=" * 60)

    method = sys.argv[1] if len(sys.argv) > 1 else "ascii"

    if method == "mermaid":
        visualize_mermaid()
    elif method == "graphviz":
        visualize_graphviz()
    elif method == "networkx":
        visualize_networkx()
    elif method == "ascii":
        visualize_ascii()
    elif method == "all":
        print("\n--- 1. Mermaid 图 ---")
        visualize_mermaid()
        print("\n--- 2. ASCII 图 ---")
        visualize_ascii()
        try:
            print("\n--- 3. NetworkX 图 ---")
            visualize_networkx()
        except Exception as e:
            print(f"NetworkX 失败: {e}")
        try:
            print("\n--- 4. Graphviz 图 ---")
            visualize_graphviz()
        except Exception as e:
            print(f"Graphviz 失败: {e}")
    else:
        print(f"未知方法: {method}")
        print("使用默认 ASCII 图...")
        visualize_ascii()
