# classification: async llm_judge, pass history and session_id

**日期**: 2026-07-09  
**提交数**: 6  

---

## 提交记录

| # | Commit | 说明 |
|---|--------|------|
| 1 | `06c84bc` | fix(classification): async llm_judge, pass history and session_id |
| 2 | `8fd966a` | refactor(classification): reorder L2/L3 — confidence pre-check before continuation |
| 3 | `1f7c35d` | refactor(classification): extract shared JSON parsing into json_utils |
| 4 | `e24ec92` | feat(classification): LLM confidence field + post-classify confidence gate |
| 5 | `683d44f` | feat(classification): BERT/embedding-based semantic continuation check |
| 6 | `c325dd1` | refactor(classification): externalize signal words to config/constants |

## 变更文件

- `agents/task_classification/classification_processor.py` (修改)
- `agents/task_classification_agent.py` (修改)
- `agents/task_classification/json_utils.py` (新增)
- `agents/task_classification/task_classifier.py` (修改)
- `tests/test_task_classification_agent.py` (修改)
- `prompts/classification_v1.txt` (修改)
- `agents/task_classification/__init__.py` (修改)
- `agents/task_classification/semantic_checker.py` (新增)
- `config/constants.py` (修改)
- `tests/test_context_lock.py` (修改)

---

## 备注

<!-- 手动补充：升级背景、设计决策、面试话术等 -->
