# memory: add layered memory skeleton (short/long-term + context mana...

**日期**: 2026-07-08  
**提交数**: 6  

---

## 提交记录

| # | Commit | 说明 |
|---|--------|------|
| 1 | `c91a518` | feat(memory): add layered memory skeleton (short/long-term + context manager) |
| 2 | `fdee2b3` | fix(memory): wire long-term memory to real embedder/store + fix 4 bugs |
| 3 | `dde0595` | feat(memory): add task memory layer (3rd tier) with structured storage |
| 4 | `5b03776` | feat(memory): add rerank, confidence filter, structured extraction |
| 5 | `c015d0d` | feat(memory): unified MemoryService + prompt externalization + docs |
| 6 | `79ea1ae` | feat(memory): conflict resolution, per-turn summary, task archive, STM sink |

## 变更文件

- `agents/context_manager.py` (新增)
- `agents/knowledge_qa_agent.py` (修改)
- `agents/memory/__init__.py` (新增)
- `agents/memory/long_term_memory.py` (新增)
- `agents/memory/short_term_memory.py` (新增)
- `tests/test_memory.py` (新增)
- `agents/environment_matching_agent.py` (修改)
- `agents/memory/task_memory.py` (新增)
- `CLAUDE.md` (修改)
- `DEV_SPEC.md` (修改)
- `agents/memory/memory_service.py` (新增)
- `api/chat_handler.py` (修改)
- `prompts/ltm_judge_and_extract_v1.txt` (新增)
- `prompts/stm_rolling_summary_v1.txt` (新增)
- `rag/ingestion/storage/chroma_store.py` (修改)

---

## 备注

<!-- 手动补充：升级背景、设计决策、面试话术等 -->
