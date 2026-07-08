# HCS Agent Platform — DEV_SPEC

> 项目级 Spec,本仓库的"宪法"。**AI 改动代码前必读**;每次任务开工前,人也要先确认 Spec 没变。
> 写于 2026-06-14(第二轮加固后)。

## 1. 目标

让测试人员(尤其新人)用自然语言完成两件事:

1. **环境条件确认**:根据测试意图自动补全环境类型 / 组件 / 区域 / 服务状态字段,筛选候选环境
2. **技术规范查询**:通过 MCP 知识库检索 SDK 文档 / 用户手册 / 内部测试规范

## 2. 非目标

- 不是生产级 SaaS(本仓库是 MVP + 简历项目;生产级 hardening 在后续 round 单独规划)
- 不替代 TestRail / Jira 等测试管理平台
- 不做多租户/多组织
- 不做云端持久化(本地 SQLite + ChromaDB 即可)

## 3. 架构

### 3.1 概念架构(4 层 + 横切支撑服务)

企业级 Agent 系统的标准分层。请求自上而下下发,结果沿原路回传,Worker↔Tool Server、Orchestrator↔Worker 支持 ReAct 多轮迭代回环。

```
┌─────────────────────────────────────────────────────────────┐
│ 1. BFF 接入网关层(统一入口)                                  │
│    认证 · 限流熔断 · 路由分发 · 基础日志 · 参数校验脱敏       │
│    + 面向前端的响应聚合/裁剪(BFF 区别于纯 API Gateway 的关键)│
├─────────────────────────────────────────────────────────────┤
│ 2. Orchestrator 智能编排层(核心调度·系统大脑)                │
│    意图识别 · 专家 Agent 路由 · 多任务串联编排               │
│    会话状态全局管理 · 流程分支决策                           │
├─────────────────────────────────────────────────────────────┤
│ 3. Worker 任务执行层(专家 Agent 运行时)                      │
│    每个 Worker 实例 = 一个专家 Agent 运行时                  │
│    LLM 推理 · 工具决策(是否调/调哪个/怎么拼参) · 沙箱隔离    │
│    Worker 是工具的主动调用方                                 │
├─────────────────────────────────────────────────────────────┤
│ 4. Tool Server 工具网关(被动能力底座·无智能)                 │
│    通过 MCP 协议统一注册/暴露/调用,兼容 Function Calling     │
│    工具注册 · 权限校验 · 协议适配 · 请求转发                 │
└─────────────────────────────────────────────────────────────┘
        │ Orchestrator → Worker → Tool Server(标准上下调用)
        │ 结果回传 + ReAct 迭代回环

横切支撑服务(被编排层/执行层依赖,非请求链路上的独立层):
  ├─ Memory 记忆服务    短期会话上下文 + 长期用户记忆 + 工作态
  ├─ Knowledge / RAG    向量库 + 混合检索(dense + BM25 + rerank)
  └─ LLM Gateway        模型路由 · 负载均衡 · Token/成本管控 · 降级回退 · 响应缓存
```

核心设计要点:

1. **层级逻辑自洽**:调度层下发任务给专家 Agent,专家 Agent 再调用工具,即 Orchestrator → Worker → Tool Server。先决策后调用工具。(网络流传的 F2 原图把 Tool Server 与 Worker 顺序写反,此处修正。)
2. **安全隔离机制**:底层任务执行依托沙箱(CPU/内存/网络/文件资源限制)+ 精细 RBAC,实现「Agent 级工具权限隔离」,不同垂直 Agent 仅可调用授权工具集;叠加 Prompt Injection 显式检测与拒答路径,以及工具入参出参的敏感信息脱敏。
3. **全链路可审计**:日志 / Metrics / Traces 三位一体。结构化 JSON 日志(输入、意图决策、工具调用、参数、结果全程可回溯可检索)+ Metrics(Token 消耗、成本、延迟、成功率)+ 分布式 Traces(trace_id 贯穿 BFF → Orchestrator → Worker → Tool Server 全链路)。日志可投喂大模型自动复盘调试。

### 3.2 实现映射(本仓库 7 层)

概念架构在本仓库的具体落地。MVP 阶段横切支撑服务尚未独立成层,能力内聚在对应模块中。

```
┌──────────────────────────────────────────┐
│ Web UI (Jinja2 + 原生 JS)                │  web/
├──────────────────────────────────────────┤
│ API (FastAPI)                            │  api/ + web/routes.py
│   ├─ require_api_key (X-API-Key)         │  api/auth.py
│   └─ SlidingWindowLimiter (10/60s)       │  api/rate_limit.py
├──────────────────────────────────────────┤
│ Agent Orchestrator (TaskClassification)   │  agents/task_classification_agent.py
│   ├─ orchestrator-worker 模式             │
│   ├─ 3 个 worker                          │
│   └─ ≤ 5 个 active agent (硬限)         │
├──────────────────────────────────────────┤
│ Specialist Agents (Worker 运行时)         │  agents/
│   ├─ TaskClassificationAgent             │  入参分类(50 golden test)
│   ├─ EnvironmentMatchingAgent             │  字段抽取+SQLite 匹配
│   └─ KnowledgeQAAgent                     │  RAG 检索 + LLM 回答
├──────────────────────────────────────────┤
│ Services (业务逻辑·横切支撑雏形)           │  services/
│   ├─ KnowledgeService → HybridSearch      │  rag/query_engine/hybrid_search.py
│   ├─ EnvironmentService                  │  候选环境筛选
│   └─ ProbeService                         │  模拟 Linux 节点探测
├──────────────────────────────────────────┤
│ Storage (Memory + Knowledge 数据底座)     │
│   ├─ ChromaDB  (dense + HNSW)            │  config/vector_store_factory.py
│   ├─ BM25Okapi (sparse + jieba)          │  rag/query_engine/sparse_retriever.py
│   └─ SQLite  (metadata + 会话快照)        │  db/
├──────────────────────────────────────────┤
│ MCP Server (Tool Server · stdio)          │  mcp_server/
│   ├─ 3 tools (query_knowledge_hub /       │
│   │  get_document_summary /               │
│   │  list_collections)                    │
│   └─ errors.py (结构化错误)               │
└──────────────────────────────────────────┘
```

概念层 → 实现层对应:
- BFF 接入网关 → `api/` + `web/routes.py`
- Orchestrator → `agents/task_classification_agent.py`
- Worker(专家 Agent 运行时) → `agents/` 下 3 个 specialist agent
- Tool Server → `mcp_server/`
- LLM Gateway 雏形 → `config/model_provider.py`(工厂模式,待补路由/限流/成本管控)
- Knowledge / RAG → `rag/` + `data/chroma/`
- Memory → `agents/memory/`(三层分层记忆,MemoryService 统一管理)
  - Short-term → `agents/memory/short_term_memory.py`(context window + rolling summary)
  - Long-term → `agents/memory/long_term_memory.py`(ChromaDB + memory gating + rerank)
  - Task → `agents/memory/task_memory.py`(结构化任务态 + 中间结果)
  - Unified → `agents/memory/memory_service.py`(按 session_id 隔离,跨 agent 共享)
  - Context assembly → `agents/context_manager.py`(token 预算 + 分层组装 + 溢出压缩)
  - Prompts → `prompts/stm_rolling_summary_v1.txt` + `prompts/ltm_judge_and_extract_v1.txt`

## 4. 关键设计决策(Why)

| 决策 | 理由 |
|------|------|
| 工厂模式覆盖 LLM/Embedding/Splitter/VectorStore/Reranker | 切 provider 不改业务代码 |
| Hybrid Search (Dense + BM25 + RRF + 可选 Rerank) | 比纯向量检索更稳;RRF 比线性加权更鲁棒 |
| SHA256 内容哈希做 doc_id | 同一文档重复入库幂等 |
| MCP 错误统一走 `errors.py` | 不把 traceback / 内部路径泄露给 LLM/Host |
| Auth + RateLimit 加在 API 层 | 防 key 被刷 + 防未授权访问 |
| Orchestrator-Worker 模式 | 避免 worker 之间互相调用导致死锁 |
| Prompts 全部在 `prompts/*.txt` | 改 prompt 不改代码,支持 A/B |
| 测试用 FakeLLM,真实 LLM 测试 skip-on-no-key | 不让 AI 自查 |

## 5. Spec 任务清单(每任务 ≤ 1 小时)

后续 round 开工前,先从这里挑一个任务,放到新 commit 标题里。

### 5.1 架构层(本仓库已有)
- [x] 工厂模式 LLM/Embedding
- [x] 工厂模式 Splitter / VectorStore / Reranker(R2-A1)
- [x] Prompts 抽到 `prompts/`(R1-Fix5)
- [x] Ingest 幂等(SHA256,R1-Fix4)
- [x] RecursiveCharacterTextSplitter(R1-Fix3)
- [x] MCP 错误结构化(R1-Fix1)
- [x] API Key auth + rate limit(R1-Fix2)
- [x] CLAUDE.md 写项目特有事实(R2-A2)
- [x] CI workflow(R2-D)

### 5.2 安全 / 输入
- [x] 输入校验 + 长度上限(R2-B)
- [ ] Prompt injection 显式检测 / 拒答路径
- [ ] 重试 / 指数退避
- [ ] Session 持久化(目前 `chats_by_session_id` 在内存,重启丢)

### 5.3 测试
- [x] E2E `/chat` 测试(R2-C)
- [x] MCP tool 单元测试(R2-C)
- [x] API auth/rate-limit 测试(R2-C)
- [ ] 覆盖率 ≥ 80%(R2-C 留 baseline)
- [ ] 数据库迁移测试(SQLAlchemy schema 升级)

### 5.4 DevX
- [x] DEV_SPEC.md(本文件)
- [x] CLAUDE.md
- [ ] Dockerfile
- [ ] docker-compose 含 ChromaDB server
- [ ] pre-commit (black + ruff + mypy)
- [ ] OpenAPI client SDK 自动生成

### 5.5 文档
- [x] README 含运行步骤
- [ ] 架构图(本文件 ASCII 草图)
- [ ] 部署文档
- [ ] 常见错误 FAQ

## 6. 文件组织约束(写代码时遵守)

```
config/             # 配置 + 工厂(LLM/Embedding/Splitter/VectorStore/Reranker/Prompt loader)
agents/             # Agent 实现(每个 agent 一个目录)
api/                # FastAPI 端点 + middleware
web/                # Jinja2 模板 + 静态资源
services/           # 业务逻辑(可被 agent 直接调用)
rag/                # RAG 检索(query_engine + ingestion + storage + embedding)
mcp_server/         # MCP 协议层
db/                 # SQLAlchemy 模型 + repositories
prompts/            # 提示词模板(.txt)
tests/              # 测试
.claude/            # (后续 round) Claude Code 规则
```

**硬约束**:
- 单文件 ≤ 200 行(写进 CLAUDE.md)
- 函数 ≤ 50 行
- 任何新增"可替换组件"必须走工厂
- 提示词独立成 `.txt`,不在 `.py` 里硬编码
- API key 不进代码、不进配置文件(dev 走 `.env`)
- 不写 `print(...)` 当日志,用 `logging.getLogger(__name__)`

## 7. 验收(Definition of Done)

每个 PR / 每个 commit 必须满足:

1. **测试全绿**:`python -m pytest tests/` 通过
2. **类型干净**:`mypy` 不新引入 error
3. **Lint 干净**:`ruff check` 无新告警
4. **不再"裸 raw exception"**:MCP/Agent 异常走 `errors.py` 包装
5. **不再"新组件无工厂"**:新增 Splitter/VS/Reranker 必须配 factory
6. **不再"新 prompt 硬编码"**:新 prompt 必须先有 `.txt` 文件
7. **commit 切细**:每 commit 完成一件小事
8. **无密钥泄露**:`grep -r "sk-" --include="*.py" .` 无结果

## 8. 跑测试

```bash
source .venv/Scripts/activate
python -m pytest tests/ -q                # 全部
python -m pytest tests/test_hybrid_search.py -v  # 单文件
python -m pytest -k "not real_llm" -q       # 跳过需要 LLM key 的(没 key 时)
```

## 9. 后续 round 候选(按价值排序)

1. 输入校验 + prompt injection 防御(已做 R2-B,还差显式拒答)
2. Session 持久化(SQLite 存 `chats_by_session_id`)
3. E2E 覆盖 + 覆盖率基线
4. Dockerfile + docker-compose
5. pre-commit
6. 真实 Reranker(目前 NoOp)

---

**变更记录**:
- 2026-06-14 v1: 初始版本,记录 R1 五项 fix + R2 A-D 计划
