# context-lock: multi-turn context lock, minimal balanced design

**日期**: 2026-07-08  
**提交数**: 2  

---

## 提交记录

| # | Commit | 说明 |
|---|--------|------|
| 1 | `ed34090` | feat(context-lock): multi-turn context lock, minimal balanced design |
| 2 | `9d3a22d` | feat(context-lock): add L3 confidence gate (multi-intent + vague clarify) |

## 变更文件

- `agents/context_lock.py` (新增)
- `agents/environment_matching/processor.py` (修改)
- `agents/task_classification/classification_processor.py` (修改)
- `agents/task_classification_agent.py` (修改)
- `prompts/context_lock_judge_v1.txt` (新增)
- `tests/test_context_lock.py` (新增)

---

## 备注

<!-- 手动补充：升级背景、设计决策、面试话术等 -->
