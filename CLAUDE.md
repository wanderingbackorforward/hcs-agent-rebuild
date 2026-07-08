# CLAUDE.md

> 项目特有事实清单,供 Claude Code / 任何 LLM agent 在本项目工作时阅读。通用编程知识不要写。

## 项目一句话

基于 `smart-appointment-ai-agent` + `MODULAR-RAG-MCP-SERVER` 二次实现的 HCS(混合云)测试辅助 Agent 平台: FastAPI + 多 Agent + MCP 知识库 + RAG Hybrid Search。

## 跑 / 测

```bash
source .venv/Scripts/activate        # Windows
python -m pytest tests/ -q           # 全测试(无 key 时真实 LLM 测自动 skip)
python -m uvicorn app:app --reload   # dev server
python -m mcp_server.server          # MCP server (stdio)
```

## 构建命令

- 包管理: `pip install -r requirements.txt`
- Python: 3.9+ (项目用 3.12.9 开发)
- 前端: Jinja2 模板 + 原生 JS,无 npm build

## 目录约束(写新文件时遵守)

- 单文件 ≤ 200 行,函数 ≤ 50 行
- 新增可替换组件(LLM / Embedding / Splitter / VectorStore / Reranker)必须走 `config/*_factory.py`
- 提示词放 `prompts/*.txt`,**不**在 `.py` 里硬编码
- MCP 工具错误统一用 `mcp_server/errors.py` 的 `MCPError` + `format_error()`,不直接 `f"Error: {e}"`
- API key 走 `.env` + gitignore,不要硬编码也不要进配置
- 输入校验在 `web/routes.py::ChatRequest` 已经有 Pydantic 校验,新加的字段必须接 `field_validator`
- **面试驱动升级**: 改代码前必须先在 `d:\mine\myprojects\interview-prep\UPGRADE_LOG.md` 追加计划记录(AOF 原则)

## 数据 / 状态

- LLM: `MODEL_PROVIDER` 切 (qwen / deepseek / minimax / openai / azure / openai-compatible)
- Embedding: `EMBEDDING_PROVIDER` 切,默认 MiniMax
- Chunker: zh 默认, 600 字符, 100 overlap,可通过 `CHUNKER_LANG` / `CHUNKER_SIZE` / `CHUNKER_OVERLAP` 覆盖
- Reranker: NoOp 默认;`CROSS_ENCODER_MODEL` 切真实 cross-encoder
- Retriever: Hybrid (Dense HNSW + BM25 + RRF)
- Session: `chats_by_session_id` 字典 + SQLite持久化(`db/repositories/session_repository.py`),重启不丢
- Auth: `API_KEYS` 设为空=dev 模式(无认证);生产必须填


## 新增模块(面试驱动升级, 2026-07-08)

- `agents/memory/` — 分层记忆(三层 + 统一服务):
  - `short_term_memory.py` — context window + rolling summary(超出窗口压缩)
  - `long_term_memory.py` — ChromaDB 向量存储 + memory gating(LLM判断重要性) + 3阶段检索(bi-encoder召回→cross-encoder rerank→置信度过滤) + 结构化实体抽取
  - `task_memory.py` — 结构化任务态 + 中间结果(存 extracted_fields["_task_memory"],无需迁移)
  - `memory_service.py` — 统一 facade,按 session_id 隔离,跨 agent 共享(注入 chat_handler)
  - Prompt 文件: `prompts/stm_rolling_summary_v1.txt` + `prompts/ltm_judge_and_extract_v1.txt`
- `agents/context_manager.py` — 上下文管理: tiktoken计数 + 分层组装(system+memory+task+conversation) + 溢出压缩策略(丢task→压STM→截断)
- `agents/knowledge_qa/react_loop.py` — ReAct循环: Thought->Action->Observation
- `cache/` — 三层Agent缓存: LLM结果缓存 + 工具结果缓存 + 语义缓存
- `code_review/` — AI Code Review: 分层验证(确定性+LLM+置信度) + 分级报告
- `mcp_server/sse_server.py` — MCP SSE transport: stdio+SSE双模式
- `tests/test_memory.py` — 33个测试覆盖 STM/LTM/TaskMemory/ContextManager(Fake组件,无API调用)
- `tests/test_agent_eval.py` — Agent评测: 任务完成率/检索精度/响应质量

完整升级日志: `d:\mine\myprojects\interview-prep\UPGRADE_LOG.md`

## 已知坑

- **MiniMax embedding 接口返回 `vectors` 顶层数组**,不是 OpenAI 兼容的 `data[].embedding`,在 `config/model_provider.py::MiniMaxEmbeddings._embed` 单独处理
- **SparseRetriever 索引在 init 时一次性建**,文档更新后必须 `sparse.refresh()`(HybridSearch.search 已自动调)
- **pytest-asyncio 严格模式**, async test 需要 `pytest-asyncio` 配置
- **MCP server stdout 是协议通道**,所有日志必须 stderr 化(`mcp_server/server.py::_redirect_all_loggers_to_stderr` 启动时处理)
- **MiniMax API 限流**:跑完整 30 对 hybrid search 测试会触发限流,本地连续跑之间 sleep 30s

## 不要做(本项目特有的禁令)

- ❌ 不要把 `.env` push 上去(已在 `.gitignore`)
- ❌ 不要在 `.py` 文件里硬编码中文 prompt,改 `prompts/*.txt`
- ❌ 不要新增 MCP tool 而不填全 6 字段 + 用 `format_error` 包装
- ❌ 不要新增"可替换组件"而绕过 factory
- ❌ 不要把测试依赖真实 LLM(用 `pytest.mark.skipif` + `_has_api_key`)
- ❌ 不要在 agent / handler 里 `f"Error: {e}"` 暴露原始异常

## Spec 引用

完整项目设计: `DEV_SPEC.md` (架构图、任务清单、验收标准)。改大方向前先读这个。

## 协作约定(commit / review)

- 每 commit 完成**一件**小事,不堆 commit
- 标题格式: `fix(#N): ...` / `feat: ...` / `refactor: ...`
- 改完跑 `pytest`,全绿才 commit
- 提交前 `git status` 检查没有意外文件
