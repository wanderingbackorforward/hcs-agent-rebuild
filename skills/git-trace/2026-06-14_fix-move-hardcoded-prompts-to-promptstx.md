# #5: move hardcoded prompts to prompts/*.txt for hot-swap

**日期**: 2026-06-14  
**提交数**: 5  

---

## 提交记录

| # | Commit | 说明 |
|---|--------|------|
| 1 | `4ea44e2` | fix(#5): move hardcoded prompts to prompts/*.txt for hot-swap |
| 2 | `11d3c81` | fix(#4): SHA256 content-hash idempotency in IngestionPipeline |
| 3 | `4f911af` | fix(#3): replace naive char-slice with RecursiveCharacterTextSplitter |
| 4 | `937d210` | fix(#1): structured MCP error envelope (no raw exception leak) |
| 5 | `5a9567e` | fix(#2): API key auth + sliding-window rate limit on /chat endpoints |

## 变更文件

- `agents/task_classification/task_classifier.py` (修改)
- `mcp_server/tools/query_knowledge_hub.py` (修改)
- `prompts/classification_v1.txt` (新增)
- `prompts/rag_answer_v1.txt` (新增)
- `rag/ingestion/pipeline.py` (修改)
- `rag/ingestion/chunking/chunker.py` (修改)
- `mcp_server/errors.py` (新增)
- `mcp_server/tools/get_document_summary.py` (修改)
- `mcp_server/tools/list_collections.py` (修改)
- `.env.example` (修改)
- `api/auth.py` (新增)
- `api/rate_limit.py` (新增)
- `web/routes.py` (修改)

---

## 备注

<!-- 手动补充：升级背景、设计决策、面试话术等 -->
