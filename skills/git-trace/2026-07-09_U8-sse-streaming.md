# [U8] SSE protocol upgrade (U8)

**日期**: 2026-07-09  
**提交数**: 4  
**分支**: `feature/sse-streaming`  

---

## 提交记录

| # | Commit | 说明 |
|---|--------|------|
| 1 | `55cd536` | feat(sse): SSE event protocol and ring buffer for structured streaming |
| 2 | `cef65ca` | feat(sse): backend SSE streaming + agent pipeline status events |
| 3 | `c523386` | feat(sse): frontend EventSource consumption with streaming display |
| 4 | `d8b7d7f` | merge: feature/sse-streaming — SSE protocol upgrade (U8) |

## 变更文件

- `api/sse_buffer.py` (新增)
- `api/sse_protocol.py` (新增)
- `config/sse_protocol.py` (新增)
- `tests/test_sse_protocol.py` (新增)
- `agents/knowledge_qa_agent.py` (修改)
- `agents/task_classification/agent_router.py` (修改)
- `agents/task_classification/classification_processor.py` (修改)
- `tests/test_context_lock.py` (修改)
- `web/routes.py` (修改)
- `web/templates/index.html` (修改)

---

## 备注

<!-- 手动补充：升级背景、设计决策、面试话术等 -->
