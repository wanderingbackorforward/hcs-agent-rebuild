# task-classification: add NLI confidence gate with degradation fallback

**日期**: 2026-07-10  
**提交数**: 1  

---

## 提交记录

| # | Commit | 说明 |
|---|--------|------|
| 1 | `cf665c7` | feat(task-classification): add NLI confidence gate with degradation fallback |

## 变更文件

- `agents/task_classification/__init__.py` (修改)
- `agents/task_classification/classification_processor.py` (修改)
- `agents/task_classification/nli_validator.py` (新增)
- `agents/task_classification_agent.py` (修改)
- `config/decision_explainer.py` (修改)
- `config/settings.py` (修改)
- `docs/intent_routing_eval_report.md` (新增)
- `eval/intent_routing_eval.py` (新增)
- `tests/test_nli_validator.py` (新增)
- `tests/test_task_classification_agent.py` (修改)

---

## 备注

<!-- 手动补充：升级背景、设计决策、面试话术等 -->
