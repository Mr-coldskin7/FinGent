# API 鉴权（访问控制）

> 对应代码：`api_auth.py`（判定逻辑）、`api_server.py`（中间件与启动校验）、`config.py`（配置）、
> `Frontend/src/utils/api.ts`（前端统一请求）、`Frontend/src/components/SettingsPanel.vue`（令牌页签）。
> 测试：`Test/test_api_auth.py`。

## 1. 为什么需要鉴权

FinGent 的后端不是只读服务。它会：

- 调用付费模型（一次多 Agent 分析约 40–60 秒，消耗额度）；
- 写入记忆数据库（反馈、纠正规则）；
- 改写 Agent 的提示词文件（`SKILL.md`）；
- 接受任意 `base_url` 并主动发起外连。

改造前，所有接口都没有鉴权。只要后端监听 `0.0.0.0` 且 `CORS_ORIGINS=*`，同一局域网里的任何机器都可以：把模型地址改成自己的服务器、让请求（含 API Key）发到那里、消耗额度、篡改 Agent 行为。

## 2. 威胁模型与缓解

| 威胁 | 例子 | 缓解 | 状态 |
|---|---|---|---|
| 局域网未授权访问 | 同网段机器调用 `PUT /api/v1/model/config` | 默认只监听 `127.0.0.1`；远程访问必须配置令牌 | ✅ 已做 |
| 凭证外泄 | 把 `base_url` 指向攻击者服务器，窃取 API Key 与用户问题 | 修改模型配置需要令牌 | ✅ 已做（`base_url` 白名单未做） |
| 额度耗尽 | 循环调用 `/api/v1/chat` | 对话与回测接口需要令牌 | ⚠️ 无速率限制 |
| 提示词篡改 | `PUT /api/v1/agents/{name}` 改写 SKILL.md | 需要令牌 | ✅ 已做 |
| 跨站请求（CSRF / 恶意网页） | 恶意页面让浏览器向后端发请求 | 令牌放在 `Authorization` 头（不是 cookie，浏览器不会自动附带）；CORS 收紧为白名单 | ✅ 已做 |
| 令牌泄露 | 写进 URL 被日志记录；截图；终端输出 | 只走请求头，不接受 query 参数；预检脚本不打印令牌内容 | ✅ 已做 |
| 页面 XSS 窃取令牌 | 模型输出里夹带 `<img onerror=…>` | 渲染 Markdown 前用 DOMPurify 清洗 | ✅ 已做 |
| 暴力猜测令牌 | 反复尝试 | 常量时间比较；令牌 ≥16 位；无失败计数与封禁 | ⚠️ 未做封禁 |

## 3. 规则：默认拒绝

所有判定集中在 `api_auth.py::decide()`，是一个**纯函数**（不依赖框架），因此可以单独测试。

**公开（无需令牌）**

| 路径 | 原因 |
|---|---|
| `GET /health` | 健康检查 |
| `GET /api/v1/market` | 公开行情，只读 |
| `GET /api/v1/eval/runs[...]` | 评测报告，只读，不含用户数据 |
| `GET /api/v1/backtest-chart` | 回测图表，只读 |
| `OPTIONS *`（CORS 预检） | 预检不携带凭证，本身不授予任何权限；真正的请求仍会被校验 |
| 非 `/api/v1/` 路径（如 `/docs`） | 只暴露接口结构，不含数据 |

**受保护（其余全部 `/api/v1/*`）**，包括：`POST /chat`、`POST /chat-stream`、`POST /feedback`、`POST /review`、`GET|PATCH|DELETE /memory/rules*`、`GET /memory/accuracy`、`GET|PUT /model/config`、`POST /model/test`、`POST /model/reset`、`POST /backtest*`、`GET|PUT /agents*`、`POST /agents/reload`。

> 新增接口默认受保护。若要公开某个只读接口，必须把它加进 `api_auth.py::PUBLIC_PATH_PATTERNS`，并在评审中说明理由。

## 4. 行为矩阵

| 是否配置 `FINGENT_API_TOKEN` | 请求来源 | 受保护接口的结果 |
|---|---|---|
| 是 | 任意 | 带正确 `Authorization: Bearer <token>` → 放行；否则 **401** |
| 否 | 回环地址（`127.0.0.1` / `::1` / `localhost`） | 放行（单机开发与演示） |
| 否 | 其他来源 | **403**，提示"服务未配置访问令牌，仅允许本机访问" |

启动时的额外检查：

- 令牌长度不足 16 位 → **拒绝启动**；
- `API_HOST` 不是回环地址且未设置令牌 → 打印警告。

## 5. 配置

**生成令牌**（至少 16 位，推荐 32 字节随机串）：

```bash
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
```

**后端**（`.env`，或启动时的环境变量，二者任选其一）：

```bash
FINGENT_API_TOKEN=<上面生成的值>
API_HOST=0.0.0.0          # 仅在需要局域网/远程访问时设置
CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173
```

> 注意：`scripts/run_backend.sh` 读取的是**启动环境变量**中的 `API_HOST`，不读取 `.env`；uvicorn 本身也不读 `.env`，但 FinGent 的配置类会读取 `.env` 中的 `FINGENT_API_TOKEN` 与 `CORS_ORIGINS`。两处要分别确认。

**前端**：打开「系统设置 → 访问令牌」，粘贴令牌，点击「验证令牌」。令牌只保存在本机浏览器的 `localStorage`（键名 `fingent.apiToken`）。

**远程访问的前提**：

1. 必须配置令牌；
2. 必须使用 HTTPS（见下文"已知局限"），否则局域网内可以嗅探到令牌；
3. 建议只允许前端的确切来源进入 `CORS_ORIGINS`。

## 6. 验证方法（curl）

```bash
TOKEN=<你的令牌>

# 公开接口：无需令牌
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8000/api/v1/eval/runs            # 200

# 受保护接口
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8000/api/v1/agents                # 401（已配置令牌时）
curl -s -o /dev/null -w "%{http_code}\n" -H "Authorization: Bearer $TOKEN" \
     http://127.0.0.1:8000/api/v1/agents                                                    # 200

# 错误的 scheme 也会被拒绝
curl -s -o /dev/null -w "%{http_code}\n" -H "Authorization: Basic $TOKEN" \
     http://127.0.0.1:8000/api/v1/agents                                                    # 401
```

验证"局域网来源被拒绝"：在未配置令牌时，用本机局域网 IP（如 `ipconfig getifaddr en0`）访问，应得到 403。

## 7. 实现要点

- **常量时间比较**：`hmac.compare_digest`，避免通过响应时间逐字节猜测令牌。
- **不放在 URL 里**：`query` 参数会进入访问日志、浏览器历史与截图，因此只接受 `Authorization` 头。
- **单一决策点**：所有放行/拒绝逻辑都在 `decide()`；中间件只负责把请求要素取出来、把结果转成 HTTP 响应。
- **统一的错误结构**：`{"success": false, "error": "…", "code": "unauthorized|forbidden"}`，401 带 `WWW-Authenticate: Bearer`。
- **前端统一出口**：所有后端请求都经过 `apiFetch()`，令牌只在一处拼装。
- **XSS 与令牌的关系**：令牌存在 `localStorage`，任何能在页面里执行脚本的人都能读取它。因此 Markdown 渲染必须清洗（`MarkdownRender.vue` 使用 DOMPurify），否则鉴权形同虚设。

## 8. 测试

`Test/test_api_auth.py`（共 29 个用例）：

- **判定表**（纯函数）：公开路径、OPTIONS 预检、受保护路径在有无令牌时的结果、错误令牌与错误 scheme、回环/远程来源、令牌长度校验。
- **HTTP 层**（TestClient）：401 + `WWW-Authenticate` 头、正确令牌放行、公开接口放行、写接口在未授权时不触库、远程来源 403、回环 200、CORS 预检不被拦截。

既有的 `Test/test_api_selfimprove.py` 测的是业务逻辑，因此固定为"关闭令牌 + 回环地址访问"，鉴权行为只在 `test_api_auth.py` 中测试。

## 9. 已知局限（必须诚实说明）

1. **单一共享令牌，没有用户身份**。`user_id` 由客户端自报，持有令牌的人可以用任意 `user_id` 写入反馈。多用户场景需要账号体系（会话或 JWT）。
2. **令牌存在浏览器 localStorage**。一旦出现 XSS 就会泄露。已通过清洗缓解，但没有 CSP 兜底；推荐后续加上 CSP，或改用 HttpOnly cookie（那需要 CSRF 防护）。
3. **明文 HTTP 下令牌可被嗅探**。远程访问必须经 HTTPS 反向代理（`nginx.conf` 目前只配了 80 端口）。
4. **没有速率限制、没有审计日志**。无法发现也无法阻止高频滥用。
5. **没有令牌轮换**。更换需要修改 `.env`、重启后端，并在前端重新粘贴。
6. **模型 `base_url` 未做白名单**。令牌持有者仍可把模型地址指向任意主机。
7. **`/docs` 与 `/openapi.json` 未受保护**。它们只暴露接口结构，不含数据；如需收紧，可在 `FastAPI(docs_url=None)` 中关闭。

## 10. 面试时可以怎么讲

> FinGent 最初是一个只在本机运行的工具，所有接口都没有鉴权，包括会消耗模型额度、会改写 Agent 提示词、会向任意地址发起外连的接口。我把它改成了默认拒绝：所有 `/api/v1` 接口都要鉴权，公开的只有只读的行情、评测和回测图表。判定逻辑集中在一个纯函数里，有 29 个单元测试和 HTTP 层测试覆盖。没配置令牌时只允许回环地址访问，配置后用常量时间比较校验 Bearer 令牌，令牌少于 16 位直接拒绝启动，CORS 收紧为白名单。前端渲染模型输出时还加了 DOMPurify 清洗，因为令牌存在 localStorage 里，XSS 会让鉴权失效。我也清楚它的局限：只有一个共享令牌，没有用户身份，没有速率限制和审计日志，远程访问必须上 HTTPS。下一步我会先补速率限制和账号体系。

## 11. 后续待办

- [ ] 速率限制（按令牌或来源 IP，对 `/chat*` 与 `/backtest*` 优先）
- [ ] 账号体系（用户 ID 由服务端签发，不再由客户端自报）
- [ ] HTTPS 反向代理配置（扩充 `nginx.conf` 的 443 端口与证书说明）
- [ ] CSP 响应头
- [ ] 模型 `base_url` 白名单与协议校验（仅允许 https）
- [ ] 审计日志（谁在何时调用了哪个写接口，不记录令牌与 API Key）
- [ ] 令牌轮换脚本
