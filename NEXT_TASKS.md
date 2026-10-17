# FinGent 任务交接（2026-10-16 晚，/clear 前落盘）

> **进度更新（2026-10-16 晚续）**：T1–T6 已全部完成并验证，详见 `Agent.md §7.9 / §8`。
> 唯一待主人决定：是否正式跑 `uv run python -m Eval.reliability --k 5`（约 25–30 分钟，消耗配额）。

> ⚠️ **注意（2026-10-16 晚）**：`Agent.md` 曾在一次写入中被截断，§1–§7.7 原文丢失，当前为依据会话重建的部分版本（见其顶部说明）。如有备份请优先恢复。

> 用途：会话清空后据此恢复上下文。决策记录以 `Agent.md`（§7 / §8）为准，本文件只记"接下来做什么 + 怎么做 + 怎么验收"。
> 恢复方式：先 `git status --short` 看改动，再读本文件，然后按顺序做 T1→T6。

## 0. 环境与当前状态

| 项 | 状态 |
|---|---|
| 后端 | `uv run uvicorn api_server:app --port 8000`（改动后必须重启才生效：`lsof -ti :8000 \| xargs kill`） |
| 前端 | `cd Frontend && npm run dev`（:5173，已完成 `npm install`） |
| 模型 | opencode zen（`MODEL_BASE_URL` / `MODEL_NAME` 见 `.env`，`x-opencode-session` header 由 `LLM/llm.py` 注入） |
| Checkpointer | `FINGENT_CHECKPOINTER=memory`（默认，无需 Redis） |
| 测试 | `uv run pytest Test -q` → 上次全量 162 passed；`Test/test_reliability.py` 单跑 19 passed |
| 前端构建 | `cd Frontend && npm run build` → 已可通过（vue-tsc + vite） |
| 未提交 | 全部改动未 commit（见 §4），需要时一并提交 |

⚠️ 全量 pytest 上次运行是在 `parse_trace` 重构**之前**；恢复后先重跑一次确认。

## 1. 今天已完成并验证（不要重做）

| # | 内容 | 验证方式 |
|---|---|---|
| D1 | `chat_node` 闲聊旁路：preprocess 提示词加 `CHAT` intent + 纯问候正则兜底，闲聊不再走 agent/投票/澄清 | API + UI 实测"你好""你是谁" |
| D2 | 全链路错误兜底：单 agent 异常→HOLD 降级；`gather(return_exceptions=True)`；`_fallback_parse` 本地兜底；`FinGraph.run` 顶层 catch；`/api/v1/chat` 300s 超时 + try/except；FastAPI 全局异常处理器 | 故障注入脚本（2 个 agent 抛异常 + 预处理解析失败）全部优雅降级 |
| D3 | Agent 角色前端管理：`GET/PUT /api/v1/agents[/{name}]`、`POST /api/v1/agents/reload`（保存→`.bak` 备份→`discover_skills()`→热重建 FinGraph）；`SettingsPanel.vue` 双 Tab（模型设置 / 🎭 Agent 角色管理） | 保存→改描述→注册表刷新→回滚，round-trip 通过 |
| D4 | SSE 实时进度：`emit_progress()`（contextvar 隔离并发）+ `POST /api/v1/chat-stream`；前端 fetch 流式解析（**注意 sse_starlette 行尾是 `\r\n`，已归一化**）；失败自动回退 `/chat`；`LoadingBubble` 显示阶段 + 秒表 + 分析师清单 | curl 全流程事件 + UI 截图 |
| D5 | 前端 `vue-tsc` 16 处错误全部修复；思维链 CHAT 文案修正 | `npm run build` 通过 |
| D6 | 同步回测循环冲突修复：`asyncio.to_thread` + 线程内先 `asyncio.run(load_price_dataframe)` 再**同步** `run_backtest`（`run_backtest` 是同步函数，不能包在 async 里） | 报错消失；但 6 个交易日 >290s，**不作为演示路径** |
| D7 | 评测模块 `Eval/reliability.py` + `Eval/cases.json` + `Test/test_reliability.py`；K 次成功率 + Wilson 区间 + p50/p95 + 降级率 + `trace.jsonl`；`--show` 渲染时间线；`Eval/runs/` 已入 `.gitignore` | 19 单测 + 真实冒烟（chat 2/2、单 agent 2/2） |
| D8 | `Agent.md` §7.1–7.8 记录上述改动 | — |

冒烟数据位置：`Eval/runs/20261016_184220/`（chat 与 single_morefit 各 2 次）。值得注意的是**单 agent 同股两次投票分别为 HOLD 与 BUY**，这是面试可用的论点（说明需要 K 次运行而非跑一次就下结论）。

## 2. 待办（按顺序）

### T1 Redis 缓存可选化，消除启动噪声 ★当前进行中（只调研了，未改代码）

现状：`FINGENT_CHECKPOINTER=memory` 时，`Data/service.py:47-54` 仍把 `_settings.redis_url` 传给 `CacheManager`，`Data/cache.py:56-65` 连接失败后 `logger.warning("CacheManager Redis 连接失败，仅使用 L1: ...")`，每个进程刷一次，演示时很难看。功能本身已降级到 L1，无正确性问题。

改法（建议）：
1. `config.py`：新增 `cache_backend: str = Field(default="memory", alias="CACHE_BACKEND")`（`memory|redis`），`REDIS_URL` 继续只服务 checkpointer。
2. `Data/service.py`：仅当 `_settings.cache_backend == "redis"` 时才传 `redis_url`，否则 `CacheManager(default_ttl=...)`。
3. `Data/cache.py`：连接失败日志降为 `logger.info`/`debug`，并明确说明"未配置 L2，使用 L1"。

验收：`FINGENT_CHECKPOINTER=memory` 启动 API，日志**不再出现**"Redis 连接失败"；`uv run pytest Test/test_cache.py -q` 通过（`Test/conftest.py:56` 用 `CacheManager(default_ttl=300)`，无 redis_url，应不受影响）。

### T2 P3 路由解释（低成本、演示可讲）

目标：前端思维链能回答"为什么选这几个 agent"。

1. `LLM/preprocess.py`：输出 schema 增加 `agent_selection_reason`（一句话），加 1–2 个示例。
2. `LLM/graph.py::_route`：`emit_progress({"stage":"route", "mode":"single|multi|chat|clarify", "agents":[...], "reason":..., "source":"explicit|semantic|fallback"})`；`_select_agents` 已有显式/语义/兜底三路判断，把来源透出来即可。
3. `api_server.py`：把 route 信息放进 `/chat` 与 `/chat-stream` 的 result 事件（如 `response["route_info"]`）。
4. 前端 `App.vue::generateThinkingSteps`：有 `route_info.reason` 时展示；`ChatResponse` 类型同步加字段。

验收：API 响应含 reason；UI 思维链显示路由理由；`pytest` + `npm run build` 通过。

### T3 前端「评测中心」

目标：把 `Eval` 的结果可视化（这是主人明确要的"前端也要体现"）。

后端（`api_server.py`，只读，注意防路径穿越）：
- `GET /api/v1/eval/runs` → 扫描 `Eval/runs/*/report.json`，按时间倒序返回摘要列表。
- `GET /api/v1/eval/runs/{run_id}` → 该目录 `report.json`。
- `GET /api/v1/eval/runs/{run_id}/trace` → 复用 `Eval.reliability.parse_trace(Path)`，返回结构化运行分组。
- `run_id` 必须正则校验 `^[0-9]{8}_[0-9]{6}$` 且目录需真实存在，禁止拼接任意路径。

前端：
- `App.vue` 的 `currentView` 联合类型加 `'eval'`，模板加分支；`Sidebar.vue` 的 `navItems` 加 `{ id: 'eval', icon: '🧪', text: '评测中心' }`（`defineProps`/`defineEmits` 的类型也要加）。
- 新组件 `Frontend/src/components/EvalPanel.vue`：运行列表 → 用例指标（能跑通/跑得对 + Wilson 区间 + p50/p95 + 降级率 + 投票分布）→ 单次运行时间线（按 stage 的横条，agent 区间用 start/end 时间戳画）。

数据前置：现在只有 K=2 冒烟。要好看的展示需先跑 `uv run python -m Eval.reliability --k 5`（3 用例 × 5 次，多 agent 约 1–2 分钟/次，合计约 25–30 分钟，消耗配额）——**跑之前问主人**。

验收：curl 三个接口；UI 能看已有冒烟数据与时间线；`npm run build` 通过。

### T4 deepagents 多余工具面

问题：`create_deep_agent` 附带 7–8 个 filesystem/todo 工具（`ls/read_file/write_file/write_todos/...`），对纯金融 agent 是多余工具面，也会干扰模型选工具。

做法：先查 `create_deep_agent` 是否支持关闭 `FilesystemMiddleware`/子中间件或传入工具白名单。可行 → 精简（但**必须保留 `read_file`**，因为 skill 加载依赖 agent 自己读 `/skills/fingent/<name>/SKILL.md`）；不可行 → 在 `Agent.md §8.4` 写明结论与原因，不再折腾。

验收：要么工具数下降且 agent 仍能正常读技能并产出 JSON；要么文档里有明确结论。

### T5 Pyright 历史警告 + 同步回测端点决策

- 决策已定：**不删** `/api/v1/backtest`，但文档标注"慢，不作为演示路径，前端只用 `/backtest-stream`"。
- Pyright：只顺手修成本低的（典型是 `Optional[str]` 的 `user_id` 传参、`new_weight` 可能未绑定）。剩下的 `dict/TypedDict`、`DataFrame` 布尔判断类噪声**不追求清零**，在 §8.4 记为已知噪声即可，避免大改引入风险。

验收：`pytest` 通过、行为无变化。

### T6 更新 Agent.md（项目约定：决策先落文档）

- **P1 / P2 不再列为待办**，记为设计限制：大模型预训练阶段已见过"未来"信息，决策准确率评测与泄漏检测在方法上就存在根本缺陷（主人判断），不作为工程任务；面试中主动披露。
- **P3** → 见 T2（实施）。
- **P4** → 主人自行调研（简化 GraphRAG 或其他新方式），Agent.md 只记计划，不由本助手实现。
- **P5** → 不做消融实验，改为从已有可靠性指标侧面体现；补记 K=2 冒烟里"同股两次投票不一致"的观察。
- **P6** → 决策：保持 LangGraph 共享 state，`trace.jsonl` 兼作审计/追踪；不上 A2A（杀鸡用牛刀）。若将来要审计，可考虑本地 md 日志 + 互斥锁。
- **工程遗留** → 补 Redis（T1 结论）、deepagents（T4 结论）、同步回测（T5）、Pyright（T5）。

验收：`Agent.md` §8 与上述一致，且每条都有"状态/结论"。

## 3. 演示前清单（面试当天）

1. 重跑：`uv run pytest Test -q`、`cd Frontend && npm run build`。
2. 三条冒烟：`你好`（闲聊旁路）、单股分析、多 Agent 投票（含进度清单）。
3. 回测演示用 **3–5 个交易日**窗口，并提前跑通一次（逐日 LLM 调用，窗口长了现场等不起）。
4. 外部依赖：LLM key 余额、Tiingo 限流、网络。
5. 若已跑 K=5，打开「评测中心」展示成功率与置信区间。

## 4. 本轮改动的文件（`git status --short` 为准）

- 后端：`api_server.py`、`LLM/graph.py`、`LLM/preprocess.py`、`config.py`（T1 待改）、`Data/cache.py` `Data/service.py`（T1 待改）
- 前端：`Frontend/src/App.vue`、`components/{LoadingBubble,SettingsPanel,ThinkingChain,BacktestChart,BacktestCandleChart,BacktestPanel}.vue`、`types/index.ts`
- 新增：`Eval/reliability.py`、`Eval/cases.json`、`Eval/__init__.py`、`Test/test_reliability.py`
- 文档：`Agent.md`、`.gitignore`

## 5. 交接注意事项

- 后端改动后**必须重启** uvicorn，否则看到的还是旧行为（曾因此误判一次）。
- `format_trace` 曾因 `parse_trace` 重构缺默认值而报 `KeyError: 'reasons'`，已修（现在 19 passed）；改 `parse_trace` 时注意保留默认字段。
- 前端新增视图时，`Sidebar.vue` 与 `App.vue` 的 `currentView` 联合类型**两边都要改**，否则 vue-tsc 报错。
- 只根据真实执行结果下结论；不要把未执行的调用当成已执行（本会话曾出现该错误）。
