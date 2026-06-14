# 对话备忘录

> 保存时间：2026-06-12
> 用途：记录当前轮对话的关键结论与下一步计划，避免丢失上下文。

---

## 一、已完成的简历工作

### 1. 万得全栈研发工程师（实习）简历
- **文件**：`D:/mine/myprojects/smart-appointment-ai-agent/简历版本/万得全栈/简历-万得-全栈研发版.md`
- **定位**：偏传统全栈，突出 React + Spring Boot + MySQL + FastAPI。

### 2. 智谱 Agent 全栈开发实习生简历
- **文件**：`D:/mine/myprojects/smart-appointment-ai-agent/简历版本/智谱Agent全栈/简历-智谱-Agent全栈开发实习生.md`
- **邮件正文**：`D:/mine/myprojects/smart-appointment-ai-agent/简历版本/智谱Agent全栈/求职邮件-智谱-Agent全栈开发实习生.md`
- **定位**：突出 AI Coding / VibeCoding、Agent memory/skill/MCP/自进化、Multi-Agent 协作、Rapid Prototyping。
- **GitHub 已加入**：https://github.com/wanderingbackorforward
- **项目链接已加入**：https://github.com/wanderingbackorforward/hcs-agent-rebuild

---

## 二、智谱岗位分析要点

- 岗位本质：**AI 应用层工程师实习版**，不是传统前后端。
- 公司意图：用 AI Coding 工具快速做产品原型、低成本试错、探索 AMiner 产品线的 Agent 化。
- 核心要求：会用 AI 写代码、能从 0 到 1 快速搭 demo、懂 Agent 技术栈（memory/skill/MCP/自进化）。
- 优势：用户项目经历匹配、熟悉 AI Coding 工具、有从 0 到 1 经验。
- 劣势：无移动端经验、专业不对口。

---

## 三、核心任务：HCS 测试辅助 Agent 平台二次实现

### 目标
基于手头的两个项目：
- `D:/mine/myprojects/smart-appointment-ai-agent`
- `D:/mine/myprojects/MODULAR-RAG-MCP-SERVER`

**二次实现 HCS 测试辅助 Agent 平台**，让用户在面试/投递时有真实可演示的项目底气。

### HCS 平台核心能力（从简历中提取）
1. **多 Agent 协作编排**：TaskClassificationAgent（意图路由）、EnvironmentMatchingAgent（环境匹配）、KnowledgeQAAgent（知识问答）。
2. **Agent Memory**：会话级上下文补全、环境条件字段持续抽取。
3. **MCP 知识检索 Server**：暴露标准化 Tool（query_knowledge_hub / list_collections / get_document_summary）。
4. **RAG / Hybrid Search**：文档摄取流水线、Dense Retrieval + BM25 + RRF + Rerank。
5. **环境匹配与实时探测**：基于结构化字段 + Linux 节点探测（host/port/component），结果写入 SQLite。
6. **TDD 与 OpenAPI**：单元测试覆盖、FastAPI 接口契约。

### 已实现模块
- [x] 配置层：多 Provider LLM/Embedding 工厂、`config/constants.py` 状态枚举
- [x] 数据层：SQLAlchemy 模型 + Repository + DatabaseRouter，含 Environment / ValidationRecord / KnowledgeDocument / UserSession
- [x] RAG 模块：IngestionPipeline、Dense Retriever、Sparse Retriever（BM25）、Hybrid Search（RRF 融合）
- [x] Agent 层：
  - `TaskClassificationAgent`：意图分类 + AgentRouter
  - `EnvironmentMatchingAgent`：多轮字段补全 + 环境匹配 + 探测落库
  - `KnowledgeQAAgent`：Hybrid Search + LLM 生成答案
- [x] MCP Server：`mcp_server/` 目录，暴露 `query_knowledge_hub` / `list_collections` / `get_document_summary`
- [x] API / Web 层：`app.py` FastAPI 入口、`web/routes.py` 聊天页、`api/chat_handler.py` 对话核心
- [x] 测试：
  - `tests/test_environment_service.py`：环境匹配与探测单元测试
  - `tests/test_task_classification_agent.py`：50 条意图路由 golden test（mock + 真实 LLM 可选）
  - `tests/test_hybrid_search.py`：30 组检索命中对
- [x] 文档：`README.md` 已更新状态清单

### 运行方式
```bash
cd D:/mine/myprojects/hcs-agent-rebuild
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
cp .env.example .env          # 填入 LLM / Embedding API Key
python app.py                 # 或 python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

访问 http://localhost:8000 使用 Web 聊天界面，http://localhost:8000/docs 查看 OpenAPI 文档。

### 注意事项
- 不配置 API Key 时，项目仍可启动，但知识库 seed 会被跳过，RAG 与 LLM 回答不可用。
- 配置 `LLM_API_KEY` 与 `EMBEDDING_API_KEY`（若使用同一 key，可只填 `LLM_API_KEY`）后，系统会自动初始化默认知识文档。
- 简历中“50 条 golden test / 85% top-10 命中率”等数字需在真实 Embedding API 环境下运行 `pytest tests/` 验证。

---

## 四、项目目录

`D:/mine/myprojects/hcs-agent-rebuild/`
