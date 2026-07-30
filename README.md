# FinGent — 智能股票分析系统

> FinGent（Financial Intelligence Agent）是一款面向个人投资者的 AI 股票分析助手。它通过多 Agent 协同投票、可解释推理链、内置回测和记忆自进化，把“这只股票能不能买”转化为可追溯、可验证、可进化的决策建议。
---

## 目录

- [FinGent — 智能股票分析系统](#fingent--智能股票分析系统)
  - [目录](#目录)
  - [项目简介](#项目简介)
  - [核心特性](#核心特性)
  - [系统架构](#系统架构)
  - [快速开始](#快速开始)
    - [1. 克隆与安装依赖](#1-克隆与安装依赖)
    - [2. 配置环境变量](#2-配置环境变量)
    - [3. 启动服务](#3-启动服务)
    - [4. 命令行模式](#4-命令行模式)
    - [5. 构建 RAG 知识库（可选）](#5-构建-rag-知识库可选)
  - [配置说明](#配置说明)
  - [目录结构](#目录结构)
  - [API 概览](#api-概览)
  - [Agent 分工](#agent-分工)
  - [开发测试](#开发测试)
  - [数据集与模型](#数据集与模型)

---

## 项目简介

FinGent 希望解决普通投资者常见的三个问题：

- **信息过载**：财报、舆情、技术面、宏观新闻混在一起，难以取舍。
- **决策黑盒**：传统选股工具只给结论，不给推理过程。
- **无法验证**：推荐买入后，历史表现如何难以量化。

为此，FinGent 设计了一个“专家委员会”式的多 Agent 系统：技术面、基本面、风控、舆情四类分析师独立调研，使用真实市场数据与金融知识库，最后通过加权投票给出统一决策。所有推理步骤、工具调用、数据来源都在前端可视化展示；任何建议都可以放进回测引擎，用历史数据验证。

---

## 核心特性

| 特性 | 说明 |
|------|------|
| **多 Agent 协同投票** | 4 类分析师并行分析，加权汇总，避免单一视角偏差 |
| **可解释推理链** | 每个 Agent 的思维链、工具调用、原始数据全部可视化 |
| **内置回测引擎** | 基于 Backtrader，支持同步回测与 SSE 流式回测 |
| **记忆与自进化** | Redis 对话记忆 + SQLite 长期记忆；用户反馈动态调整 Agent 权重 |
| **RAG 金融知识库** | 本地 ChromaDB + Qwen Embedding，支持金融术语与指标检索 |
| **A 股 / 美股双市场** | 自动识别市场类型，akshare / yfinance / FMP / Tiingo 多源适配 |
| **国产模型优先** | 默认通义千问，数据不出境，支持多模型切换 |

---

## 系统架构

FinGent 采用六层架构，层间通过明确的数据契约交互：

```
用户请求
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  ① 用户交互层 (Vue3 + TypeScript)                        │
│     职责: 零代码交互、可视化展示、SSE 流式接收            │
├─────────────────────────────────────────────────────────┤
│  ② API 网关层 (FastAPI)                                  │
│     职责: 请求调度、接口标准化、CORS、参数校验            │
├─────────────────────────────────────────────────────────┤
│  ③ 核心引擎层 (LangGraph)                                │
│     职责: 任务编排、状态流转、多轮对话、并行计算          │
├─────────────────────────────────────────────────────────┤
│  ④ Agent 智能体层 (LLM + Tools)                          │
│     职责: 多 Agent 独立分析、工具调用、CoT 追踪           │
├─────────────────────────────────────────────────────────┤
│  ⑤ 数据服务层 (Data Service)                             │
│     职责: 市场识别、路由、获取、标准化、缓存              │
├─────────────────────────────────────────────────────────┤
│  ⑥ 回测引擎层 (Backtrader)                               │
│     职责: 历史模拟、绩效评估、审计日志、流式推送          │
└─────────────────────────────────────────────────────────┘
    │
    ▼
持久化层: Redis (L1 对话记忆) + SQLite (L3 长期记忆) + ChromaDB (RAG 向量库)
```

一条典型请求的生命周期：

```
用户输入: "分析一下贵州茅台"
    │
    ▼ ① 前端 POST /api/v1/chat
    │
    ▼ ② FastAPI 参数校验
    │
    ▼ ③ FinGraph.run() 启动状态机
    │       ├── preprocess: 提取 "600519" / "stock_analysis"
    │       ├── route: 判断路由到 "ALL" (多 Agent 并行)
    │       ├── ALL 节点: 并发 4 个 Agent
    │       │       ├── TECHNICAL_NERD  → 投票 BUY
    │       │       ├── Morefit         → 投票 STRONG_BUY
    │       │       ├── RiskManager     → 投票 HOLD
    │       │       └── SentimentAnalyzer → 投票 BUY
    │       ├── voting: 加权汇总
    │       └── END: 输出结果
    │
    ▼ ④ 记忆系统记录分析历史
    │
    ▼ ⑤ FastAPI 封装响应
    │
    ▼ ⑥ 前端渲染: 决策卡片 + 思维链 + 工具调用追踪
```

---

## 快速开始

### 1. 克隆与安装依赖

```bash
# 后端依赖（建议使用 Python 3.9+）
pip install -r requirements.txt

# 前端依赖
cd Frontend
npm install
```

> 若项目暂未提供 `requirements.txt`，可直接安装关键依赖：`fastapi`、`uvicorn`、`langgraph`、`langchain`、`akshare`、`yfinance`、`backtrader`、`redis`、`aiosqlite`、`chromadb`。

### 2. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env`，至少填写 `QIANWEN_API_KEY`。其他数据源 API Key 按需填写。

### 3. 启动服务

```bash
# 启动 Redis（如使用本地 Redis）
redis-server

# 启动后端 API
python api_server.py

# 启动前端（新终端）
cd Frontend
npm run dev
```

默认后端地址：`http://localhost:8000`  
默认前端地址：以 Vite 输出为准（通常为 `http://localhost:5173`）

### 4. 命令行模式

```bash
python main.py
```

### 5. 构建 RAG 知识库（可选）

```bash
python RAG/build_db.py
```

---

## 配置说明

所有配置通过 `.env` 管理，核心变量如下：

| 变量名 | 必填 | 默认值 | 说明 |
|--------|------|--------|------|
| `QIANWEN_API_KEY` | 是 | - | 阿里云百炼 API 密钥 |
| `MODEL_BASE_URL` | 否 | `https://dashscope.aliyuncs.com/compatible-mode/v1` | 大模型 API 地址 |
| `MODEL_NAME` | 否 | `qwen-max` | 模型名称 |
| `REDIS_URL` | 否 | `redis://localhost:6379/0` | Redis 连接串 |
| `FINGENT_MEMORY_DB` | 否 | `./memory/l3_memory.db` | L3 记忆数据库路径 |
| `FMP_API_KEY` | 否 | - | Financial Modeling Prep 密钥 |
| `TIINGO_API_KEY` | 否 | - | Tiingo 密钥 |
| `BOCHA_API_KEY` | 否 | - | Bocha AI Search 密钥 |
| `API_HOST` | 否 | `0.0.0.0` | 后端监听地址 |
| `API_PORT` | 否 | `8000` | 后端端口 |

完整配置见 [`config.py`](config.py)。

---

## 目录结构

```
FinGent/
├── api_server.py              # FastAPI 主服务
├── main.py                    # CLI 入口
├── config.py                  # 统一配置管理
├── backtest_cli.py            # 回测命令行工具
│
├── Data/                      # 数据服务层
│   ├── cache.py               # LRU 内存缓存
│   ├── memory.py              # L3 记忆管理器
│   ├── models.py              # 统一数据模型
│   ├── service.py             # InfoService 统一入口
│   ├── unified_models.py      # 跨市场通用模型
│   └── providers/             # 数据提供商适配器
│       ├── adapters.py
│       ├── us_stock.py        # 美股数据源
│       ├── zh_stock.py        # A 股数据源
│       └── web_search.py      # 网络搜索
│
├── LLM/                       # Agent 与核心引擎层
│   ├── base.py                # Agent 基类
│   ├── graph.py               # FinGraph 状态机
│   ├── agent.py               # 四类 Agent 实现
│   ├── router.py              # 意图路由
│   ├── preprocess.py          # 输入预处理
│   ├── tools.py               # 基础工具定义
│   ├── unified_stock_tools.py # 统一股票工具
│   ├── risk_tools.py          # 风险计算工具
│   └── sentiment_tools.py     # 舆情分析工具
│
├── RAG/                       # RAG 金融知识库
│   ├── build_db.py
│   └── db_operations.py
│
├── Trade/                     # 回测引擎层
│   ├── runner.py
│   ├── adapter.py
│   ├── visualizer.py
│   └── backtest_cli.py
│
├── Frontend/                  # 用户交互层（Vue3 + Vite + Tailwind）
│   ├── package.json
│   ├── vite.config.ts
│   └── src/
│       ├── App.vue
│       ├── types/
│       └── components/
│           ├── MarketPanel.vue
│           ├── BacktestPanel.vue
│           ├── MessageBubble.vue
│           ├── DecisionCard.vue
│           ├── ThinkingChain.vue
│           └── ToolChain.vue
│
├── Test/                      # 单元测试
└── memory/                    # SQLite 持久化数据库
```

---

## API 概览

| 方法 | 端点 | 说明 |
|------|------|------|
| POST | `/api/v1/chat` | 对话分析 |
| POST | `/api/v1/feedback` | 用户反馈 |
| GET  | `/api/v1/market` | 行情聚合 |
| POST | `/api/v1/backtest` | 同步回测 |
| POST | `/api/v1/backtest-stream` | SSE 流式回测 |
| POST | `/api/v1/backtest-cancel` | 取消回测 |
| GET  | `/api/v1/backtest-chart` | 回测图表数据 |
| POST | `/api/v1/review` | 历史决策复盘 |
| GET  | `/health` | 健康检查 |

详细请求/响应示例见 [`FinGent_技术方案书_v1.md`](FinGent_技术方案书_v1.md)。

---

## Agent 分工

| Agent | 角色 | 核心能力 | 代表工具 |
|-------|------|---------|---------|
| **TECHNICAL_NERD** | 技术面分析师 | 量价分析、散庄博弈、筹码分布 | `get_stock_price`、`search_financial_knowledge` |
| **Morefit** | 基本面分析师 | Mini-Business 拆解、安全边际估值 | `get_stock_financial_statements`、`get_stock_company_info` |
| **RiskManager** | 风控分析师 | VaR、波动率、最大回撤、凯利仓位 | `calculate_var`、`calculate_max_drawdown`、`kelly_position_sizing` |
| **SentimentAnalyzer** | 舆情分析师 | 新闻情绪、市场温度、逆向信号 | `search_news_sentiment`、`analyze_market_sentiment` |

每个 Agent 输出统一格式：

```json
{
  "vote": "BUY | HOLD | SELL | STRONG_BUY | STRONG_SELL | REDUCE",
  "target_position_pct": 0.6,
  "confidence": 0.85,
  "reason": "详细分析理由...",
  "symbol": "600519"
}
```

---

## 开发测试

```bash
# 运行测试
pytest Test/

# 前端类型检查与构建
cd Frontend
npm run build
```

---

## 数据集与模型

- 金融知识数据集：[deepseek-fin](https://www.modelscope.cn/datasets/jwangkun/deepseek-fin)
- Embedding 模型：[Qwen/Qwen3-Embedding-0.6B](https://www.modelscope.cn/models/Qwen/Qwen3-Embedding-0.6B/summary)

---

