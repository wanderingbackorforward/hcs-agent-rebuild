# cache: wire three-layer caching across RAG pipeline

**日期**: 2026-07-09  
**提交数**: 1  

---

## 提交记录

| # | Commit | 说明 |
|---|--------|------|
| 1 | `4bf2730` | feat(cache): wire three-layer caching across RAG pipeline |

## 变更文件

- `.env.example` (修改)
- `agents/knowledge_qa/react_loop.py` (修改)
- `agents/knowledge_qa_agent.py` (修改)
- `agents/memory/long_term_memory.py` (修改)
- `agents/memory/short_term_memory.py` (修改)
- `agents/task_classification/task_classifier.py` (修改)
- `cache/__init__.py` (新增)
- `cache/llm_cache.py` (新增)
- `cache/registry.py` (新增)
- `cache/semantic_cache.py` (新增)
- `cache/tool_cache.py` (新增)
- `config/settings.py` (修改)
- `mcp_server/prompts/classify_intent.py` (修改)
- `mcp_server/prompts/rag_answer.py` (修改)
- `mcp_server/resources/prompt_catalog.py` (修改)
- `mcp_server/tools/query_knowledge_hub.py` (修改)
- `prompts/__init__.py` (新增)
- `prompts/loader.py` (新增)
- `prompts/react_v1.txt` (新增)
- `rag/query_engine/hybrid_search.py` (修改)
- `rag/query_engine/sparse_retriever.py` (修改)
- `services/knowledge_service.py` (修改)
- `tests/test_memory.py` (修改)
- `tests/test_prompt_cache.py` (新增)
- `tests/test_semantic_cache.py` (新增)
- `tests/test_tool_cache.py` (新增)

---

## 备注

<!-- 手动补充：升级背景、设计决策、面试话术等 -->
