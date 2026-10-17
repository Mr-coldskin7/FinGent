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
| **国产模型优先** | 默认通义千问（OpenAI 兼容接口），`.env` 可切换任意兼容模型或中转（如 opencode zen），支持多模型切换 |

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

### 1. 安装依赖（uv 环境）

```bash
# 后端依赖（Python 3.12，由 uv 自动管理）
uv sync

# 可选依赖组：RAG 向量库（torch/chromadb）、回测（backtrader）、开发测试
uv sync --group rag
uv sync --group backtest
uv sync --group dev

# 前端依赖
cd Frontend
npm install
```

### 2. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env`，至少填写 `QIANWEN_API_KEY`。其他数据源 API Key 按需填写。

### 3. 启动服务

```bash
# 启动后端 API（默认内存 checkpointer，无需 Redis）
uv run uvicorn api_server:app --port 8000

# 如需 Redis 持久化对话记忆（L1）：先启动 Redis，再设置环境变量
redis-server
FINGENT_CHECKPOINTER=redis uv run uvicorn api_server:app --port 8000

# 启动前端（新终端）
cd Frontend
npm run dev
```

`FINGENT_CHECKPOINTER` 取值：`memory`（默认，进程内存）/ `redis`（需要 Redis 运行中）。详见 [`Agent.md`](Agent.md)。

默认后端地址：`http://localhost:8000`  
默认前端地址：以 Vite 输出为准（通常为 `http://localhost:5173`）

### 4. 命令行模式

```bash
uv run python main.py                          # 交互式模式
uv run python main.py "分析AAPL值得买吗"      # 单次查询
```

### 5. 构建 RAG 知识库（可选，需先 uv sync --group rag）

```bash
uv run python RAG/build_db.py
```

---

## 配置说明

所有配置通过 `.env` 管理，核心变量如下：

| 变量名 | 必填 | 默认值 | 说明 |
|--------|------|--------|------|
| `QIANWEN_API_KEY` | 是 | - | 阿里云百炼 API 密钥 |
| `MODEL_BASE_URL` | 否 | `https://dashscope.aliyuncs.com/compatible-mode/v1` | 大模型 API 地址 |
| `MODEL_NAME` | 否 | `qwen-max` | 模型名称 |
| `REDIS_URL` | 否 | `redis://localhost:6379/0` | Redis 连接串（L1 对话记忆 / L2 行情缓存） |
| `FINGENT_CHECKPOINTER` | 否 | `memory` | 对话记忆后端：`memory`（进程内存）/ `redis` |
| `CACHE_BACKEND` | 否 | `memory` | 行情缓存后端：`memory`（仅 L1 内存）/ `redis`（L2 Redis） |
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
│   ├── cache.py               # L1 内存缓存 + L2 Redis 缓存（CACHE_BACKEND 可选）
│   ├── memory.py              # L3 记忆管理器（权重反馈 / 纠正规则 / 结果统计）
│   ├── outcome.py             # 决策结果前向验证（超额收益判定 → 客观权重）
│   ├── market_context.py      # 板块/市场问答用的实时背景行情
│   ├── symbol_alias.py        # 显式股票别名表（如 海力士 → SKHY）
│   ├── models.py              # 统一数据模型
│   ├── service.py             # InfoService 统一入口
│   ├── unified_models.py      # 跨市场通用模型
│   └── providers/             # 数据提供商适配器
│       ├── adapters.py
│       ├── us_stock.py        # 美股数据源
│       ├── zh_stock.py        # A 股数据源
│       ├── yahoo_us.py        # Yahoo 直连行情（Tiingo 兜底源）
│       ├── ak_isolated.py     # akshare 子进程隔离层（防原生层崩溃）
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
├── Eval/                      # 可靠性评测
│   ├── reliability.py         # K 次运行成功率 / Wilson 区间 / 延迟 / 降级率
│   ├── cases.json             # 评测用例
│   └── runs/                  # 每次评测产物（report / trace / run_log，不入库）
│
├── scripts/
│   └── run_backend.sh         # 后端守护脚本（崩溃自动拉起）
│
├── Frontend/                  # 用户交互层（Vue3 + Vite + Tailwind）
│   ├── package.json
│   ├── vite.config.ts
│   └── src/
│       ├── App.vue
│       ├── types/
│       └── components/
│           ├── MarketPanel.vue     # 行情中心
│           ├── BacktestPanel.vue   # 回测
│           ├── MessageBubble.vue
│           ├── DecisionCard.vue    # 决策卡片
│           ├── ThinkingChain.vue   # 思维链（含路由理由 / 板块问答标记）
│           ├── LoadingBubble.vue   # SSE 进度 + 阶段秒表
│           ├── EvalPanel.vue       # 评测中心（成功率 / 时间线 / 运行日志）
│           ├── RulesPanel.vue      # 规则管理
│           ├── SettingsPanel.vue   # 模型设置 + Agent 角色管理
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
| POST | `/api/v1/chat-stream` | 对话分析（SSE 实时进度） |
| POST | `/api/v1/feedback` | 用户反馈（多 Agent 反馈归因 → 权重） |
| GET  | `/api/v1/market` | 行情聚合（stale-while-revalidate 缓存） |
| POST | `/api/v1/backtest` | 同步回测（较慢，演示路径请用 stream 版） |
| POST | `/api/v1/backtest-stream` | SSE 流式回测 |
| POST | `/api/v1/backtest-cancel` | 取消回测 |
| GET  | `/api/v1/backtest-chart` | 回测图表数据 |
| POST | `/api/v1/review` | 历史决策复盘（投票拆解 / 结果验证 / 反转提醒） |
| GET/PUT | `/api/v1/agents[/{name}]` | Agent 角色管理（描述可编辑，热重载） |
| GET  | `/api/v1/eval/runs[/{id}[/trace|/log|/report-md]]` | 可靠性评测产物（只读） |
| GET/PATCH/DELETE | `/api/v1/memory/rules[/{id}]` | 纠正规则管理 |
| GET  | `/api/v1/memory/accuracy` | Agent 命中率 |
| GET/POST | `/api/v1/model/config|test|reset` | 模型运行时配置 |
| GET  | `/health` | 健康检查 |

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
uv run pytest Test -q

# 可靠性评测（K 次运行成功率，消耗 LLM 配额，3 用例 × 5 次约 25–30 分钟）
uv run python -m Eval.reliability --k 5          # 全部用例
uv run python -m Eval.reliability --k 2 --only chat   # 冒烟

# 决策结果前向验证（超额收益 → 客观权重更新）
uv run python -m Data.outcome

# 可选：结果验证定时调度（launchd，工作日 18:10 自动跑）
cp scripts/com.fingent.outcome.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.fingent.outcome.plist

# 前端类型检查与构建
cd Frontend
npm run build
```

---

## 数据集与模型

- 金融知识数据集：[deepseek-fin](https://www.modelscope.cn/datasets/jwangkun/deepseek-fin)
- Embedding 模型：[Qwen/Qwen3-Embedding-0.6B](https://www.modelscope.cn/models/Qwen/Qwen3-Embedding-0.6B/summary)

---

