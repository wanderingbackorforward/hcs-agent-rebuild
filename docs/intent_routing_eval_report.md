# 意图路由准确率评估报告

> 评测时间: 2026-07-10 | 评测模式: live (真实 LLM) | Golden Set: 55 条

## 总体指标

| 指标 | 值 |
|------|------|
| 总用例数 | 55 |
| 正确路由数 | 52 |
| 错误路由数 | 3 |
| **准确率** | **94.5%** |

## 各意图召回率

| 意图 | 总数 | 正确 | 召回率 |
|------|------|------|--------|
| environment_match | 25 | 25 | 100.0% |
| knowledge_qa | 25 | 22 | 88.0% |
| unrelated | 5 | 5 | 100.0% |

## 混淆矩阵 (行=期望, 列=预测)

| 期望 \ 预测 | environment_match | knowledge_qa | unrelated |
|---|---|---|---|
| environment_match | 25 | 0 | 0 |
| knowledge_qa | 3 | 22 | 0 |
| unrelated | 0 | 0 | 5 |

## 错误案例分析

| # | 查询 | 期望意图 | 实际意图 | 置信度 |
|---|------|---------|---------|--------|
| 1 | region 参数怎么填 | knowledge_qa | environment_match | 0.90 |
| 2 | 准备阶段需要什么 | knowledge_qa | environment_match | 0.85 |
| 3 | 组件必须处于 available 状态吗 | knowledge_qa | environment_match | 0.90 |

## 错误模式分析

3 个错误全部为 **knowledge_qa → environment_match 误判**，共同特征：

- 查询中包含环境领域关键词（`region`、`准备阶段`、`available`）
- 但实际意图是询问技术规范，而非匹配环境
- LLM 被关键词误导，置信度偏高（0.85-0.90）

### 改进方向

1. **Prompt 优化**: 在 `classification_v1.txt` 中增加边界判例，明确"包含环境关键词但询问规范/要求时归为 knowledge_qa"
2. **关键词冲突处理**: `region`、`available` 同时出现在 ENV_SIGNAL 和知识问答场景中，可考虑在 prompt 中加入消歧规则
3. **低置信度兜底**: 3 个误判置信度均为 0.85-0.90，高于当前阈值 0.5，可针对 knowledge_qa 场景的边界 case 调整阈值或增加二次确认

## 测试架构

### 四层测试体系

| 层级 | 测试内容 | 需要 API Key | 用例数 |
|------|---------|-------------|--------|
| Layer 1: JSON 解析 | `parse_classification_json` 处理所有意图标签 | 否 | 55 |
| Layer 2: FakeLLM 端到端 | 分类器 + JSON 管道与 stub LLM 协作 | 否 | 3 |
| Layer 3: 指标逻辑 (replay) | 准确率/召回率/混淆矩阵计算逻辑 | 否 | 7 |
| Layer 4: 真实 LLM 准确率 | 全量 55 条跑真实 LLM，断言 >= 90% | 是 | 2 |

### 评估指标

- **准确率** (Accuracy): 正确路由数 / 总测试数 = 52/55 = 94.5%
- **各意图召回率** (Recall): 该意图正确数 / 该意图总数
- **混淆矩阵**: 展示期望意图与实际意图的交叉分布

### 评估模块

- `eval/intent_routing_eval.py` — 评估核心模块
- `tests/test_task_classification_agent.py` — 四层测试
