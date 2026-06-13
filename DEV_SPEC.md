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

## 3. 架构(7 层)

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
│ Specialist Agents                         │  agents/
│   ├─ TaskClassificationAgent             │  入参分类(50 golden test)
│   ├─ EnvironmentMatchingAgent             │  字段抽取+SQLite 匹配
│   └─ KnowledgeQAAgent                     │  RAG 检索 + LLM 回答
├──────────────────────────────────────────┤
│ Services (业务逻辑)                       │  services/
│   ├─ KnowledgeService → HybridSearch      │  rag/query_engine/hybrid_search.py
│   ├─ EnvironmentService                  │  候选环境筛选
│   └─ ProbeService                         │  模拟 Linux 节点探测
├──────────────────────────────────────────┤
│ Storage                                   │
│   ├─ ChromaDB  (dense + HNSW)            │  config/vector_store_factory.py
│   ├─ BM25Okapi (sparse + jieba)          │  rag/query_engine/sparse_retriever.py
│   └─ SQLite  (metadata + 会话快照)        │  db/
├──────────────────────────────────────────┤
│ MCP Server (stdio)                        │  mcp_server/
│   ├─ 3 tools (query_knowledge_hub /       │
│   │  get_document_summary /               │
│   │  list_collections)                    │
│   └─ errors.py (结构化错误)               │
└──────────────────────────────────────────┘
```

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
