# HCS 测试辅助 Agent 平台

> 二次实现版：基于 `smart-appointment-ai-agent` 的多 Agent 架构与 `MODULAR-RAG-MCP-SERVER` 的 RAG + MCP 能力，从零搭建一个可运行的 HCS（混合云）测试辅助 Agent 平台。

---

## 项目目标

让测试人员（尤其是新人）通过自然语言对话完成：
1. **环境条件确认**：根据测试意图自动补全环境类型、组件需求、服务状态等字段，筛选候选环境。
2. **技术规范查询**：通过 MCP 化的知识库检索 SDK 文档、用户手册、内部测试规范。

---

## 核心能力

| 能力 | 说明 |
|------|------|
| 多 Agent 协作 | TaskClassificationAgent → EnvironmentMatchingAgent / KnowledgeQAAgent |
| Agent Memory | 会话级上下文补全，持续抽取环境条件字段 |
| MCP 知识检索 Server | 暴露 `query_knowledge_hub` / `list_collections` / `get_document_summary` |
| Hybrid Search | Dense Embedding + BM25 稀疏检索 + RRF 融合 + 可选 Rerank |
| 文档摄取流水线 | Load → Split → Transform → Embed → Upsert |
| 环境匹配 | 结构化字段筛选 + 模拟 Linux 节点探测 + SQLite 验证快照 |
| TDD / OpenAPI | 单元测试覆盖 + FastAPI 自动生成接口文档 |

---

## 技术栈

- **Web 框架**：FastAPI + Uvicorn
- **AI 框架**：LangChain
- **LLM/Embedding**：OpenAI-compatible API（支持 Qwen / DeepSeek / Zhipu / Azure）
- **向量库**：ChromaDB
- **稀疏检索**：自研 BM25
- **数据库**：SQLite + SQLAlchemy
- **缓存/限流**：Redis（可选，用于分布式限流和 LLM 结果缓存）
- **MCP 协议**：mcp SDK（stdio）
- **前端**：Jinja2 + 原生 JS（可选）

---

## 目录结构

```
hcs-agent-rebuild/
├── app.py                          # FastAPI 应用入口
├── requirements.txt
├── .env.example
├── README.md
├── config/                         # 配置层
│   ├── model_provider.py           # 多模型 Provider 工厂
│   ├── database.py                # 数据库与 Redis 配置
│   ├── constants.py
│   └── settings.py
├── db/                             # 数据持久层
│   ├── models.py
│   ├── base.py
│   ├── db_router.py
│   └── repositories/
├── agents/                         # Agent 智能层
│   ├── task_classification_agent.py
│   ├── task_classification/
│   ├── environment_matching_agent.py
│   ├── environment_matching/
│   └── knowledge_qa_agent.py
│       └── knowledge_qa/
├── services/                       # 业务服务层
│   ├── environment_service.py
│   ├── probe_service.py
│   └── knowledge_service.py
├── mcp_server/                     # MCP 知识检索 Server
│   ├── server.py
│   ├── protocol_handler.py
│   └── tools/
├── rag/                            # RAG 检索与摄取
│   ├── ingestion/
│   └── query_engine/
├── api/                            # API 层
│   └── chat_handler.py
├── web/                            # Web 层
│   ├── routes.py
│   ├── templates/
│   └── static/
└── tests/                          # 测试
```

---

## 快速开始

### 1. 安装依赖

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入 LLM API Key 与 Base URL
```

### 3. 运行

```bash
python app.py
```

访问 http://localhost:8000/docs 查看 API 文档。

---

## 当前状态

- [x] 项目骨架
- [x] 配置层
- [x] 数据层
- [x] RAG 模块
- [x] MCP Server
- [x] Agent 层
- [x] API / Web 层
- [x] 测试
- [x] 文档
- [ ] Redis 集成（分布式限流、LLM 结果缓存）
