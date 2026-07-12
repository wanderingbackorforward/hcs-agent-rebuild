# mcp: add Resource and Prompt primitives to complete MCP trio

**日期**: 2026-07-09  
**提交数**: 1  

---

## 提交记录

| # | Commit | 说明 |
|---|--------|------|
| 1 | `7c6844a` | feat(mcp): add Resource and Prompt primitives to complete MCP trio |

## 变更文件

- `mcp_server/__init__.py` (修改)
- `mcp_server/prompts/__init__.py` (新增)
- `mcp_server/prompts/classify_intent.py` (新增)
- `mcp_server/prompts/rag_answer.py` (新增)
- `mcp_server/protocol_handler.py` (修改)
- `mcp_server/resources/__init__.py` (新增)
- `mcp_server/resources/knowledge_catalog.py` (新增)
- `mcp_server/resources/prompt_catalog.py` (新增)
- `mcp_server/resources/server_status.py` (新增)
- `tests/test_mcp_resources_prompts.py` (新增)

---

## 备注

<!-- 手动补充：升级背景、设计决策、面试话术等 -->
