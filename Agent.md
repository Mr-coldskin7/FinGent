# FinGent Agent.md（部分重建版）

> ⚠️ **事故说明（2026-10-16）**：本文件在一次脚本写入中被截断为 0 字节，原文（§1–§6、§7.1–§7.7 完整内容，以及 §8 的原始表述）已丢失。本文件是依据会话记录重建的，只覆盖 §7.8 起的本轮工作。
> **如果你有备份**（Time Machine、编辑器本地历史、其它副本），请以备份为准，并用备份恢复全文后再合并本节内容。
> 当前交接状态另见 `NEXT_TASKS.md`。

---

### 7.8 可靠性评测模块（2026-10-16，对应 P1 的最小可行部分）

- 位置：`Eval/reliability.py`（+ `Eval/cases.json`，`Test/test_reliability.py`）。
- 命令：`uv run python -m Eval.reliability --k 5`（全部用例）；`--only chat,single_morefit` 冒烟；`--show <trace.jsonl>` 渲染时间线。
- 指标：**能跑通**（无异常 + 状态合法 + 有结果）与 **跑得对**（再加路由符合预期、无降级 agent、投票合法）两级；每个用例给出 Wilson 95% 置信区间、p50/p95 延迟、降级率、投票分布、失败原因去重计数。
- 边界：只衡量可靠性，不衡量决策准确率。

### 7.9 本轮收尾（T1–T6）

- **T1 缓存可选化**：新增 `CACHE_BACKEND`（默认 `memory`），仅 `redis` 时才连接 Redis；连接失败日志降为 info。
- **T2 路由解释**：`_select_agents` 返回 `(names, source)`，source ∈ `explicit / semantic / fallback`；`_route` 产出 `route_info {mode, agents, reason, source}` 并推送 `stage=route`；preprocess 新增 `agent_selection_reason`；`/chat` 与 `/chat-stream` 响应带 `route_info`；前端思维链展示理由。
- **T3 评测中心**：只读接口 `GET /api/v1/eval/runs`、`/runs/{id}`、`/runs/{id}/trace`；`run_id` 正则校验，解析后的目录必须位于 `Eval/runs` 下（非法 400、不存在 404）。
- **T4 deepagents 工具面**：`create_deep_agent` 无关闭开关，改用 `AgentMiddleware.wrap_model_call` + `ModelRequest.override(tools=...)` 在模型请求层剔除 `ls / write_file / edit_file / glob / grep / execute / write_todos / task`，**保留 `read_file`**（技能是渐进式披露，agent 需要用它读 SKILL.md）。实测过滤前 9 个工具、过滤后 2 个；真实单 Agent 仍能读技能并输出 JSON。
- **T5**：反馈接口 `user_id` Optional 兜底、`new_weight` 显式初始化；`/api/v1/backtest` 标注为同步慢接口，前端只用 `backtest-stream`。
- **T6**：本节与 §8 同步。

### 7.10 K=5 可靠性评测结果与 review

- 产物：`Eval/runs/20261016_194013/`。K=5，3 个用例，共 15 次。
- 指标：chat 5/5、single_morefit 5/5、multi_nvda 5/5 能跑通且跑得对；95% Wilson 区间 [0.57, 1.00]（样本太小，下界只有 0.57）。延迟 p50/p95：chat 3.4s/3.9s，single 42s/68s，multi 44s/58s（n=5 时 p95 即最大值）。
- 投票波动（真实发现）：single_morefit（茅台）HOLD×4 / BUY×1；multi_nvda（英伟达）最终投票 HOLD×1 / STRONG_BUY×2 / BUY×2；分 Agent：technical-nerd HOLD×3/BUY×2，risk-manager HOLD×4/BUY×1，sentiment-analyzer HOLD×2/BUY×3，morefit BUY×5。
- **review 发现的测量漏洞（已修）**：`_parse_agent_output` 在 JSON 解析失败时静默返回 HOLD/0.5，不带任何错误标记，导致"降级率"与"跑得对"无法发现解析失败。修复：解析失败打 `parse_fallback=True`，`evaluate_run` 将其计为降级。新增 3 个单测。
- **局限**：上述报告生成于修复之前，当时没有保存每个 Agent 的原始输出，因此无法判断解析失败发生过几次。**需要用修复后的代码重跑 K=5 才能给出可靠结论。**「跑得对」只表示路由正确、无异常、无降级、投票合法，不代表结论正确（没有真值）。

### 7.11 结构化评测日志 + 前端报告/日志区块（2026-10-16 晚）

- 每个批次新增 `run_log.jsonl`（JSON Lines，固定字段 `ts / level / event`），事件为 `suite_start / run_start / run_end / suite_end`。`run_end` 带每个 Agent 的 `vote / error / parse_fallback`。level：跑不通=ERROR，跑通但不合格=WARN，其余=INFO。
- 接口：`GET /api/v1/eval/runs/{id}/log[?level=]`（坏行计入 `skipped_lines`）、`GET /api/v1/eval/runs/{id}/report-md`。两者在批次缺少对应文件时返回 200 与 `available: false`（早期批次没有结构化日志，属正常情况）。
- 前端评测中心：报告原文（默认收起，`MarkdownRender` 渲染）与结构化运行日志（级别过滤、点击行展开）。

### 7.12 行情加速 + 深色看盘风格（2026-10-16 晚）

- **加载慢的原因**：`/api/v1/market` 每次全量拉取（A 股 AKShare 快照约 24s，美股 Tiingo 串行 8s）；`zh_stock.recent_stock_list` 还会把整张表 `print` 到控制台。
- **修复**：后端行情缓存采用 stale-while-revalidate（zh TTL 60s、us TTL 120s；过期先返回旧数据并后台刷新；响应带 `cached / age_s / stale`）；启动时后台预热；美股改为 `asyncio.gather` 并发；去掉 `print`；外部调用加 60s 超时。前端 `MarketPanel` 的缓存提升到模块级，切换页面回来仍然立即显示。实测行情首屏由约 24s / 8s 降到约 57ms（缓存命中）。
- **视觉**：整体改为纯黑底（`#000`）、深灰面板、琥珀色强调、红涨绿跌（沿用 A 股习惯）；去掉光球与网格装饰；删除机器人、火箭等装饰 emoji，功能性状态改用 ✓ ✕ ! … 等文字符号；欢迎页去掉光圈，改为细琥珀色分隔线。
- **未改动**：聊天 LLM 分析本身的耗时（约 40–60s，由模型决定，由 LoadingBubble 展示进度）；模型输出内容中的 emoji；协议匹配文本（`## 📊 双Agent综合分析`，与前端判断逻辑保持一致）。

---

## 8. 剩余待办（按面试/演示优先级）

### 8.1 演示前必须（已完成）
- 闲聊旁路、全链路兜底、角色前端管理、SSE 实时进度、前端 vue-tsc 全绿、回测流式可用、评测中心、行情缓存。

### 8.2 演示前建议
- 用修复后的代码重跑 `uv run python -m Eval.reliability --k 5`，再看评测中心。
- 面试前跑一次 `pytest Test/` + `npm run build` + 三条冒烟（"你好"、单股分析、多 Agent 投票）。
- 回测演示固定 3–5 个交易日窗口，并提前跑通一次。
- 外部依赖预热：LLM key 余额、Tiingo 限流、网络。

### 8.3 历史未完成项（状态）
| # | 问题 | 当前状态 |
|---|------|---------|
| P1 | 决策准确率评测缺失 | 设计限制：可靠性评测见 §7.8；准确率需真值与 point-in-time 数据，大模型预训练已见过"未来"信息，不作为工程任务，面试中主动披露 |
| P2 | 回测数据泄漏 | 设计限制，同 P1，面试中披露 |
| P3 | LLM Router 偏弱 | 已实施（§7.9 T2）：语义选择 + 理由 + route_info 透出到思维链 |
| P4 | 向量库偏弱 | 主人自行调研（简化 GraphRAG 或其它方式），助手不实现 |
| P5 | Agent 数量无证据 | 不做消融实验；改从可靠性指标体现，并记录 K 次运行的投票波动（§7.10） |
| P6 | Agent 通信靠共享 state | 决策：保持 LangGraph 共享 state，不上 A2A；`trace.jsonl` 兼作审计 |
| P7 | Prompt/工具耦合 | 已完成（skills）+ 前端可编辑 |

### 8.4 工程遗留
- ✅ deepagents 工具面（§7.9 T4）。
- ✅ 同步回测（§7.9 T5）；Pyright 仅修低成本项，其余历史类型噪声（`agent: dict = None`、`add_conditional_edges` 类型、SSE ping 工厂类型、`review` 段 `user_id` Optional、NDFrame 布尔判断）记为已知噪声。
- ✅ Redis 启动噪声（§7.9 T1）。
- ✅ 结构化评测日志（§7.11）、行情缓存（§7.12）。
- ⚠️ 本文件 §1–§7.7 原文丢失，待恢复（见顶部事故说明）。

### 7.13 演示稳定性：后端崩溃与"network error"兜底（2026-10-16 晚）

- **事故**：akshare 的 `stock_zh_a_spot` 内部使用 py_mini_racer（V8），在 API 进程线程池中并发调用时触发原生层 FATAL，整个 uvicorn 进程退出，前端显示裸的 "network error"。
- **修复（分层）**：
  1. 所有 akshare 调用经 `Data/providers/ak_isolated.py` 在 spawn 子进程池执行；子进程崩溃只影响该次调用，池子损坏时自动重建并重试一次，仍失败则抛出中文 `RuntimeError`。`zh_stock.py` 的 9 处 `asyncio.to_thread(ak.X)` 已全部改为 `ak_call`。
  2. `scripts/run_backend.sh` 守护进程：uvicorn 退出后 2 秒内自动拉起（实测杀进程后 4 秒恢复），日志写入 `logs/api.log`。
  3. 前端统一经 `Frontend/src/utils/errors.ts` 的 `friendlyError()` 翻译错误：网络失败显示"后端服务暂时无响应，请稍后重试…"，超时、异常数据各有提示；聊天、行情、评测、设置页均已接入。
  4. 行情：空结果（数据源限流）视为失败，不再缓存为成功；最近一次成功的行情落盘到 `.cache/market/`，后端重启后立即可用（实测 6ms）。
- **验证**：隔离层三种场景（正常、调用中途崩溃、池子已坏）均返回数据；杀后端进程后 4 秒恢复；后端全停时行情页显示中文提示；`pytest` 166 passed；`npm run build` 通过。
- **已知外部依赖**：Tiingo 免费额度按小时限流（本次测试消耗完毕，美股暂时返回明确错误）；A 股冷启动约 50 秒，之后由快照与缓存提供。
- **演示前操作**：启动后端用 `./scripts/run_backend.sh`；提前访问一次行情中心，让 A 股快照生成。

### 7.14 美股行情兜底：Yahoo Finance 直连（2026-10-16 夜）

- **数据流**：Tiingo 为主源；Tiingo 报错、被限流、抛异常或**返回不完整**（少于 20 只）时，整体切到 Yahoo Finance，不混用两个源。响应带 `source`（`tiingo` / `yahoo`），前端行情页标注"数据源 …"。
- **Yahoo 接入**：`Data/providers/yahoo_us.py` 调用 v8 chart 接口（无需 key，直连实测 200、约 0.14 秒/只，20 只并发约 0.5 秒）。v7 quote 接口需要 crumb，直连 401，因此不用。解析为纯函数 `parse_chart`，便于样本单测。
- **格式差异（已归一化）**：Yahoo 返回的是 K 线 + meta，不是逐条报价。
  - 昨收：最后一根 K 线若与 `regularMarketTime` 同日（今日已有 K 线），取倒数第二根；否则最后一根即昨收。否则涨跌会差一天。
  - 名称：用 `shortName`（Tiingo 只能用代码代替）。
  - 市值：chart 接口不提供，显示为 `-`（Tiingo 路径同样没有市值）。
- **发现并修复的缺陷**：
  1. Tiingo 抛异常时 `_get_us_market` 未捕获，异常会穿出成 500 → 已改为进入兜底。
  2. Tiingo 限流时部分请求静默缺票（实测只返回 10/20），原逻辑只在"完全为空"时兜底 → 改为"不完整即兜底"。
  3. 启动时加载的残缺快照被当作有效数据 → 美股快照必须完整才加载。
- **测试**：`Test/test_yahoo_us.py`（6 个，固定样本：今日有/无 K 线、缺值、零昨收、无名称）；`Test/test_us_market_fallback.py`（5 个，模拟 Tiingo 限流/残缺/异常/两源皆挂）。全量 pytest 通过。
- **实测**：直连 Yahoo 20/20；真实接口在 Tiingo 部分限流时自动切到 Yahoo，冷启动 2.6 秒，之后缓存命中。浏览器行情页 20 行，无报错。
- **额度提醒**：每次美股刷新会向 Tiingo 发 20 次请求。Tiingo 免费额度按小时计，若 120 秒 TTL 下持续有人访问，每小时约 600 次，会超出额度。演示前如需稳定，可将 `_MARKET_TTL_S["us"]` 调大（例如 600），或直接把 Yahoo 设为主源（仍保留 Tiingo 兜底）。

### 7.15 无个股实体不再澄清：板块/市场问答（2026-10-16 夜）

- **问题**："科技股还能买吗"被识别为 SUGGESTIONS，但没有个股实体，`_route` 直接走 `need_stock_name` → `clarify_node`，前端显示"需要更多信息"。这是死胡同：板块问题本来就不对应单一股票，不应该强制解析个股。
- **修复**：
  1. `_route`：无实体时改走新节点 `sector_node`（route_info `mode=sector`，`source=sector`），不再澄清。
  2. `_fallback_parse`：LLM 解析失败且提不出代码时，不再返回 `clarification_needed`，改为 `SUGGESTIONS` + 无实体，同样走板块回答。
  3. `_sector`：若问题涉及科技/美股/AI/芯片等，先取实时背景（`Data/market_context.py`，Yahoo 直连一篮子 14 只科技股，缓存 120 秒）；模型只能引用背景中的数字，数据不可用时明确禁止编造；提示词要求说明"还能不能买"没有统一答案、不构成投资建议，并引导用户给出股票名称或代码再做多 Agent 分析；单日涨跌不得推断资金流向或趋势。模型失败时有固定兜底文案。
  4. 前端：思维链新增"板块/市场问题 → 直接回答"；`sector` 进度事件、加载气泡文案、类型定义同步。评测 `classify_route` 识别 `sector`。
- **保留的澄清**：LLM 明确判定需要澄清（如"这个怎么样"指代不清）时，仍走 `clarify_node`，并非所有澄清都取消。
- **验证**：`Test/test_sector_route.py` 9 个用例（路由、兜底解析、节点在有/无数据/无模型时的行为、数字来源、评测分类）；全量 pytest 186 通过；`npm run build` 通过。真实接口"科技股还能买吗"返回 `mode=sector`，回答中的 14 只篮子统计与行情一致（涨 6 跌 8，平均 +0.27%）；前端页面复测思维链与回答正常，无澄清气泡。

### 7.16 自我改进 P0–P2 落地 + 存储决策（2026-10-16 深夜）

**存储决策：保持 SQLite，不换 Postgres/Redis。**
- 理由：单机本地应用，写入量每天几十条，无多实例、无并发写压力；换库只增加运维，解决不了现在的问题（问题在数据语义与闭环，不在引擎）。
- 改为：`PRAGMA user_version` 逐版迁移（当前 v3），投票拆为 `agent_votes` 行表（可直接按 Agent 查询），新增 `agent_outcomes` 结果表。
- 触发换库的条件（记在这里，满足再评估）：需要多进程/多机部署；写入量达到每秒级；需要跨用户的大规模分析查询。

**P0 修复**
- **权重口径统一**：有效权重 = 客观权重(`global` 行，来自前向验证) × 个人调整(用户行，来自反馈)，截断到 [0.3, 1.5]。投票与反馈使用同一口径。此前反馈写用户行、投票读 `global`，反馈从未影响投票。
- **复盘接口** `POST /api/v1/review`：补上前端已在调用的路由（此前 404）。返回当前决策、理由、投票拆解、反对的 Agent、结果验证、近 5 条历史、方向反转提醒。
- **多 Agent 反馈归因**：`agent_names` 由前端勾选传入；不勾选只记录历史、不改权重。
- **不一致检测修复**：此前在写入本次记录之后查询，总是和自己比较，方向反转告警永远不会触发。现在排除本次记录 id。
- **时间统一为 UTC**：SQLite `CURRENT_TIMESTAMP` 是 UTC，截止时间原用本地时间，相差 8 小时。
- **Agent 名称统一**：历史数据里有 `TECHNICAL_NERD`、`RiskManager`、`Morefit` 等写法，同一 Agent 会被拆成多份权重。统一为注册表的标准名（`technical-nerd`、`morefit`、`risk-manager`、`sentiment-analyzer`），迁移 v3 改写历史行。

**P1 结果验证（前向检验）**
- `Data/outcome.py`，命令：`uv run python -m Data.outcome`（`--no-weights` 只验证）。
- 入场价 = 决策日之后第一个共同交易日的**开盘价**；出场价 = 入场日起第 N 个交易日的收盘价。决策时刻之后的价格不参与入场，无前视偏差；代价是偏保守（决策日盘中走势不计入）。
- 判定用**超额收益**（个股 - 基准）：美股基准 `^GSPC`，A 股 6 位代码基准 沪深300。BUY/STRONG_BUY 超额 > +2% 为对；SELL/REDUCE 超额 < -2% 为对；HOLD 在 ±2% 内为对。
- 数据不足或未到期记为"待验证"，不误判。
- 客观权重 = clamp(1 + 2 × (命中率 − 1/3), 0.3, 1.5)，1/3 为三分类随机基线。**每个 Agent 至少 30 个已验证样本才更新**（`MIN_VERIFIED_SAMPLES`）。
- 主要验证期 20 个交易日（`PRIMARY_HORIZON`），5 日结果同时保存。

**P2 治理**
- **测试隔离**：`Test/conftest.py` 在导入前把 `FINGENT_MEMORY_DB` 指向临时库。此前 `api_server` 导入时会创建指向真实库的全局记忆管理器，测试数据会写进真实库（历史上的污染来源）。
- **数据清理**（已备份至 `memory/backups/`，该目录已加入 `.gitignore`）：删除 2026-05-13 测试窗口内的 46 条分析、14 条统计、8 条规则（用户名均为 test/fb/conc 等测试标识）。保留 `anonymous` 下 12 条（含 2026-09-22 与 2026-10-16 记录）。**2026-10-16 的几条很可能是今天的测试调用，需要你确认是否删除。**
- **规则管理接口**：`GET /api/v1/memory/rules`、`PATCH /api/v1/memory/rules/{id}`（启用/禁用）、`DELETE /api/v1/memory/rules/{id}`，只能操作自己的规则。**前端规则管理界面尚未实现。**
- **不一致提醒**：在复盘结果中展示。
- 命中率接口：`GET /api/v1/memory/accuracy?horizon=20`。

**已知局限（必须对外披露）**
- 目前每个 Agent 只有 2 个 20 日样本，远低于门槛，**权重不会动**。这是设计使然，不是故障。积累到 30 个样本需要较长时间。
- 6 位数字代码一律视为 A 股（`.SS`/`.SZ`）。海力士（`000660`）已在 §7.17 映射到美股 SKHY；港股、其他韩股暂不支持（用户不看）。
- 验证需要定期运行（cron/launchd），目前是手动命令，未自动调度。
- 判定阈值 2% 与 20 日窗口是经验值，未做敏感性分析。
- 多 Agent 勾选反馈的界面只做了构建验证，未用真实多 Agent 分析做端到端测试（一次分析约 60 秒、消耗模型额度，且会写入真实库）。

**测试**：全量 pytest 通过（新增 `test_memory_selfimprove.py`、`test_outcome.py`、`test_api_selfimprove.py`）；`npm run build` 通过。

### 7.17 海力士映射到美股 SKHY + 规则管理界面（2026-10-17 凌晨）

**海力士 → SKHY（不再使用韩股）**
- 用户不看韩股/港股。SK 海力士在美股以 SKHY（NASDAQ，SK hynix Inc.）上市，Yahoo 直连可用，上市日 2026-07-10。
- 此前：`us_stock_cik_map.csv` 已有 `SK海力士→SKHY`，但 `service.py` 注释还写着过期的 HXSCL（Yahoo 返回 404）；"海力士"单独、`000660` 都不命中，会落入 A 股分支。
- 现在：新增 `Data/symbol_alias.py` 显式别名表（海力士 / SK海力士 / SKHYNIX / HYNIX / 000660 → SKHY），`us_stock.input2Symbol` 与结果验证 `yahoo_symbol / benchmark_for` 共用。别名一律基准标普500，不落到沪深300。
- **记录保留**：历史记录 51（`000660`，2026-09-21）未改写，验证时经别名映射为 SKHY，5 日结果已写入（HOLD 判定为对，超额 -1.62%）。
- `Test/test_symbol_alias.py`：14 个用例。

**规则管理界面**
- 侧边栏新增「规则管理」（`Frontend/src/components/RulesPanel.vue`，主视图 `rules`，传入当前用户 ID）。
- 功能：列表（规则内容、适用 Agent 中文名、触发次数、来源、创建时间）、状态开关（启用/禁用）、删除（二次确认）、按状态筛选、统计卡（总数/已启用/强规则）。强规则阈值与后端 `WEIGHT_RULE_PROMOTE_THRESHOLD` 一致（3）。
- 不提供手动新增：规则只能从分析结果的「纠正规则」反馈产生，保证每条规则有可追溯的反馈来源。
- 浏览器实测：列表、禁用（DB 状态同步）、筛选、删除（取消/确认两步）均正常；截图 `.playwright-mcp/rules_panel.png`。测试规则与测试用户的个人权重行已清理。

**同时修复的一个真实缺陷（规则从未注入）**
- 反馈规则按标准名存储（`morefit`），而 Agent 注入时传的是类名（`Morefit`，来自 `self.__class__.__name__`），查询不到 → 所有纠正规则从未进入提示词。
- 修复：`add_or_update_rule / get_rules` 入口统一为标准名。新增两个测试：类名可取到标准名规则；禁用规则不注入。
- 旧测试 `test_get_rules` 的期望值同步更新为标准名。

**测试**：全量 pytest 253 通过；`npm run build` 通过。
**未做**：规则的"自动挖掘"来源仍未实现（`source=self_mined` 只是占位）；定时运行结果验证仍是手动命令。

### 7.18 收尾与 review 修复（2026-10-17 上午，两批提交）

**第一批：安全 checkpoint（35f6cdf）+ 立即项**
- 把累计 81 个未提交文件（约 1.9 万行）提交为 checkpoint；`.playwright-mcp/`、`memory/*.db`、`*.bak` 加入 `.gitignore`；`l3_memory.db` 与 `SKILL.md.bak` 移出 git 追踪（review P1：live 用户数据库不应入库）。**残留**：历史提交里仍有旧版 db（副防御已做不了，提醒主人若仓库公开需注意）。
- `.gitattributes` 把 `*.md`/`*.json` 标 binary 的旧行删除（只锁 lockfile），README/Agent.md 恢复正常 diff。
- README 刷新：删除不存在的技术方案书引用；补全评测/规则/复盘/Agent 管理等新端点；目录结构补 Eval/、scripts/、yahoo_us、ak_isolated、symbol_alias 等；新增 Eval.reliability / Data.outcome / launchd 调度命令；模型描述修正为"OpenAI 兼容接口可切换"。
- K=5 重跑（修复 parse_fallback 后）：`Eval/runs/20261017_112055/`，15/15 能跑通且跑得对，降级率 0%，解析降级 0 次；投票波动仍在（morefit HOLD×7/BUY×3 等），单股两次结果不一致的问题已被 K 次运行机制覆盖。
- 结果验证定时调度：`scripts/run_outcome.sh` + `scripts/com.fingent.outcome.plist`（工作日 18:10），安装命令见 README。
- 反馈权重双倍叠加 bug（review P1）：`/api/v1/feedback` 曾同时调 `adjust_weight` + `record_agent_outcome`，同一 delta 应用两次。修复：`record_agent_outcome` 返回最终权重，端点只调它一次；新增量值回归测试（agree +0.05、disagree −0.10）。

**第二批：review P2/P3 修复 + 功能项**
- **规则自动挖掘（原占位兑现，任务 #5）**：`MemoryManager.mine_outcome_rules()`——从 `agent_outcomes` 判错行（主验证期）提炼候选纠正规则，纯模板不调 LLM；新规则 `active=0`、`source=self_mined`，同一身份的重复失败仅 trigger_count 自增；已启用的规则重挖时不会被降级。接入 `Data.outcome.run()`（输出 `self_mined_rules`）。规则面板的 `self_mined` 来源标签已预留。6 个新单测（`Test/test_self_mined_rules.py`）。
- **美股主源互换（任务 #6）**：Yahoo 直连为主源，Tiingo 降为兕底（免费额度按小时限流，持续访问会超额；Yahoo 无 key 无配额）。两源都不完整时报中文错误并注明两侧原因。`_get_us_market` 双层兕底均不能穿透为 500。实测 Yahoo 主源 20/20。
- **单 Agent 决策误记 HOLD（review P2）**：`graph.run` 归档时单 Agent 无 `final_decision`，曾归为 HOLD。现在从唯一投试点导出真实投票（与 Eval 口径一致）；`final_decision` None 时不再误用 `.get`。
- **验证队头阻塞（review P2）**：schema v4 新增 `analysis_history.verify_attempts`；已过期仍取不到行情的记录每次记账一次，达 `MAX_VERIFY_ATTEMPTS=10` 后不再拉取，避免退市/数据源故障的旧行永远占据 oldest-200 名额。
- **L2 缓存失效缺口（review P2）**：`CacheManager.invalidate(symbol)` 现在同时删 Redis 键（维护 symbol→key 映射），防止 L1 过期后旧数据"复活"。
- **选做 P3**：`run_backend.sh` 尊重 `API_HOST/PORT` + 退避重启；规则 `created_at` 统一 UTC；RulesPanel 补 tiktok-auntie 标签；`Eval/cases.json` 增加 sector 用例；`fetch_daily_bars` 的 period2 揪 +1 天（开区间）；graph/api_server 的 fire-and-forget 任务保留引用；无个股实体（闲聊/板块）不再写入分析历史；`CACHE_DEFAULT_TTL` 接入 `InfoService`（此前是死配置）；`Trade/backtest_audit.jsonl` 入 gitignore。
- **遗留（review 未修）**：`_get_us_market` 全有或全无（19/20 也整批兕底）— 保持一致性优先，后续可考虑部分接受阈值；pyright 历史噪声按 §8.4 维持不追零。


### 7.19 演示前整理（2026-10-17）

**死代码与重复（已删除）**
- `Data/unified_models.py`：与 `Data/models.py` 逐行相同（仅注释不同），无引用 → 删除。
- `Trade/backtest_cli.py`：2026-05 旧版副本，与根目录 `backtest_cli.py`（README 首推、功能更新）重复 → 删除，根目录版为唯一回测 CLI。
- `Frontend/src/components/BacktestChart.vue`：无引用（回测图表用 `BacktestCandleChart.vue`）→ 删除。
- `Frontend/public/test-chart.html`：开发期接口测试页，直接请求后端 → 删除。
- `Data/memory.py::_sync_record_feedback`：无引用 → 删除。
- `config.py` 的 `mysql_url`、`memory_db_path`：无引用（记忆库直接读 `FINGENT_MEMORY_DB`）→ 删除；`.env.example` 同步去掉 `MYSQL_URL`。
- `LLM/base.py`、`LLM/preprocess.py`、`LLM/router.py` 末尾的 MySQL `__main__` 演示块及其内嵌坏测试（`PyMySQLSaver` 未导入时必然报错）→ 删除。
- `api_server.py` 的通配导入 `from LLM.unified_stock_tools import *`：未使用 → 删除。

**命名统一**
- Python 类 `TECHNICAL_NERD` → `TechnicalNerd`（其余三个已是驼峰）。运行时投票键仍是注册表名，类名只参与记忆注入，而记忆层的 `canonical_agent_name` 忽略大小写与分隔符，两种写法都归一到 `technical-nerd`（已验证）。
- 前端 Agent 名称/颜色映射原有 5 处独立维护，且键使用 `Morefit / TECHNICAL_NERD` 等类名，**而后端实际发送的是 `morefit / technical-nerd`，映射从未命中**（界面显示原始英文 ID、颜色落到默认灰）。现统一到 `Frontend/src/utils/agents.ts`，键为注册表名，查找时归一化。

**重复逻辑合并**
- FinGraph 组装原在 `main.py`、`api_server.py`（主图与两处回测图）各写一遍 → 合并为 `LLM/graph.py::assemble_fin_graph(llm, checkpointer, memory_manager, agent_names)`。回测仅传入 `BACKTEST_AGENT_NAMES = ("technical-nerd", "morefit")`，行为与原先一致。

**代码质量**
- 新增 `[tool.ruff]` 基线（`E9 / F / B`，忽略 B905）。改动前 859 条（大量为未配置时的风格类噪声），改动后该基线下 **0 条**。
- 修复：未使用导入/变量、重复导入、循环变量、缺少异常链（B904，补 `from err`）。
- `@app.on_event("startup")` → `lifespan`，消除 FastAPI 弃用告警（测试告警 4 → 2）。
- pyright 历史噪声（HEAD 时即存在）仍按 §8.4 维持，不追零。

**缓存与产物**
- 删除 `__pycache__`、`.pytest_cache`、`.ruff_cache`、`.playwright-mcp/`（测试截图，已 gitignore）。
- `git rm --cached` 了 63 个被追踪的 `.pyc`（此前 `.gitignore` 已忽略但未清出索引）。
- **保留** `.cache/market/`：启动即用的行情快照，演示时可把美股/A 股首屏从 24–50 秒降到毫秒级；`Eval/runs/`、`memory/`、`memory/backups/` 为演示证据与用户数据，不删。

**本轮发现的真实缺陷**
- `Data/providers/web_search.py` 读取的环境变量名为 `BO_CHA_API_KEY`（多一个下划线），而 `.env` 中为 `BOCHA_API_KEY`，导致请求头恒为 `Bearer None`、联网搜索恒 401。已修正；未配置 key 时改为明确报错 `未配置 BOCHA_API_KEY`，不再发出非法请求头。**注意：当前 `.env` 中 `BOCHA_API_KEY` 为空，新闻检索仍不可用，需主人决定是否配置。**
- `scripts/run_backend.sh` 默认 `API_HOST=0.0.0.0`，uvicorn 不读取 `.env`，即使 `.env` 写了 127.0.0.1 也不生效 → 脚本默认值改为 `127.0.0.1`。需要局域网访问时显式设置 `API_HOST=0.0.0.0`。

**演示支持**
- `scripts/demo_preflight.py`：只用标准库；检查 `.env` 键是否存在（不打印值）、后端、Agent 注册表、模型配置、行情预热、前端构建、评测产物、pytest；`--ask` 发一次真实问答。
- `DEMO.md`：演示脚本（每步的耗时与讲解点）、已知风险与应对、兜底方案、禁忌操作。

**验证**
- `uv run pytest Test -q` → 261 passed（2 warnings）。
- `uv run ruff check .` → All checks passed。
- `Frontend: npm run build` → 通过（vue-tsc + vite）。
- 后端重启后：preflight `--ask` 全部通过；单 Agent 分析（morefit）返回 `route_info.mode=single`、`agent_name=morefit`；浏览器完整跑通一次多 Agent 投票，决策卡片标签已显示为中文名。
- 反馈勾选框与思考链标题的标签替换仅经构建验证，未在浏览器中重新跑一次分析确认。

**未做（有意留到演示之后）**
- `api_server.py`（约 1700 行）按 router 拆分；`Data/memory.py` 拆分。改动面大，演示前不动。
- `requirements.txt`（UTF-16 编码）与 `pyproject.toml` / `uv.lock` 并存，待确认是否保留。
- 旧版 `LLM/tools.py` 仍被 preprocess 用于 `fuzzy_search_us_symbols`，与 `unified_stock_tools.py` 功能部分重叠，待统一。

### 7.20 API 鉴权 + Markdown 清洗（2026-10-17 晚）

- **决策**：默认拒绝。`/api/v1` 下除只读公开接口（行情、评测报告、回测图表）外，全部需要鉴权。判定逻辑集中在 `api_auth.py::decide()`（纯函数，29 个测试）。
- **行为**：配置 `FINGENT_API_TOKEN`（≥16 位，否则拒绝启动）后，受保护接口需要 `Authorization: Bearer`；未配置时仅回环地址可访问，其它来源 403。
- **配置变更**：`API_HOST` 默认改为 `127.0.0.1`；`CORS_ORIGINS` 默认改为前端本地地址（`.env` 中的 `*` 需人工改）；CORS 允许头收紧为 `Authorization, Content-Type`。
- **前端**：所有请求经 `Frontend/src/utils/api.ts::apiFetch()`（23 处调用已替换）；设置面板新增「访问令牌」页签（保存在 localStorage，可验证、可清除）。
- **安全缺口的同步修复**：`MarkdownRender.vue` 原先用 `v-html` 直接渲染模型输出，没有清洗。令牌存在 localStorage，XSS 会使鉴权失效，因此引入 DOMPurify（已用 jsdom 验证 onerror / script / javascript: 链接被去除）。
- **验证**：pytest 290 passed；ruff 通过；`npm run build` 通过；真实进程验证——本机无令牌 200；0.0.0.0 + 令牌实例，局域网无令牌 401、错误令牌 401、Basic scheme 401、正确令牌 200、公开接口 200、CORS 预检 200；0.0.0.0 无令牌实例，局域网 403、本机 200。
- **未验证**：前端在"已配置令牌"模式下的完整交互（当前前端访问的是本机后端，始终免令牌）。
- **文档**：`docs/AUTH.md`（威胁模型、规则、行为矩阵、配置、测试、局限、面试表述、待办）。
- **未做（已列入 AUTH.md 待办）**：速率限制、账号体系、HTTPS 反代配置、CSP、base_url 白名单、审计日志、令牌轮换。
- **RAG**：本轮只写了待讨论的计划 `docs/RAG_TODO.md`，未实施。RAGAS 判断：现阶段不需要，先做确定性指标。
