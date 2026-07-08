# HCS Agent 评估框架

> 5 核心指标 · 离线 + 线上双轨评测

评估一个 Agent，本框架看五个核心指标，同时用**离线** + **线上**两套方式评测：

| # | 指标 | 看什么 | 离线评测 | 线上评测 |
|---|------|--------|----------|----------|
| 1 | **任务成功率** | 最后事有没有做对（看结果） | golden set 跑期望结果匹配 | 滚动窗口成功率统计 |
| 2 | **回答质量** | 输出准不准确、有没有幻觉（看输出） | 参考答案重叠 + 幻觉检测（LLM-judge 可选） | 回答长度/幻觉安全度采样 |
| 3 | **执行轨迹** | 过程有没有乱调用、死循环、乱结束（看过程） | trace 回放 + 模式检测 | trace 采样 + 异常告警 |
| 4 | **工具调用** | 工具选得对不对、参数对不对（看能力） | 工具选择准确率 + 参数 schema 校验 | 调用错误率/冗余率监控 |
| 5 | **延迟和 Token** | 速度快不快、成本高不高（看工程落地） | benchmark 基准 P50/P95 + token 成本 | 实时 P50/P95 + 成本面板 |

---

## 设计思路

### 核心数据结构：`AgentTrace`

五个指标全部读自同一个 `AgentTrace`，把**采集**和**打分**解耦：

```
AgentTrace
├── query / final_answer / success / termination_reason   ← 指标 1, 2
├── steps[]  (Thought / Action / Observation)              ← 指标 3
│   └── tool_call (name / args / result / success)         ← 指标 4
├── latency_ms / prompt_tokens / completion_tokens         ← 指标 5
└── retrieved_context (用于幻觉检测)                        ← 指标 2
```

### 两种采集模式

- **Live 模式**：`TraceRecorder` 挂到 `ReActLoop`，实时记录每一步。需要 LLM API key。
- **Replay 模式**：喂入预录/合成的 `AgentTrace`，**无需任何 API key**，CI 直接跑。

---

## 目录结构

```
eval/
├── trace.py              # AgentTrace / TraceStep / ToolCallRecord / TraceRecorder（公共数据基础）
├── metrics_content.py    # 指标 1 任务成功率 + 指标 2 回答质量/幻觉检测
├── metrics_process.py    # 指标 3 执行轨迹 + 指标 4 工具调用
├── metrics_cost.py       # 指标 5 延迟 P50/P95 + Token 成本
├── metrics.py            # 聚合入口 compute_all_metrics()（5 指标 + 综合分）
├── golden_cases.py       # 离线 golden 集（6 标注用例 + 回放 trace + 工具 schema）
├── offline.py            # 离线评测器：跑 golden 集 → 算 5 指标 → 出报告
├── online.py             # 线上评测器：FastAPI 中间件 + /eval/metrics /eval/traces
├── report.py             # Markdown 报告渲染
└── README.md             # 本文件

prompts/
├── eval_llm_judge_v1.txt       # LLM-as-judge 提示词（可选，增强指标 2）
└── eval_hallucination_v1.txt   # 幻觉检测提示词（可选，增强指标 2）

tests/
└── test_eval_framework.py      # 19 个测试，replay 模式，无需 API key
```

---

## 快速使用

### 离线评测（无需 API key，CI 友好）

```python
from eval.offline import OfflineEvaluator

ev = OfflineEvaluator(mode="replay")   # 或 "auto"（有 key 走 live，无 key 走 replay）
report = ev.run()
print(report["markdown"])              # 打印 Markdown 报告
print(report["overall_score"])         # 综合分 0..1
```

### 线上评测（挂到 FastAPI）

```python
from eval.online import attach_to_app

attach_to_app(app, sample_rate=0.1)    # 采样 10% 的 /chat 请求
# GET /eval/metrics  → 滚动窗口 5 指标实时快照
# GET /eval/traces   → 最近 N 条 trace 明细
```

### 给 Agent 挂 TraceRecorder（live 模式采集）

```python
from eval.trace import TraceRecorder
from agents.knowledge_qa.react_loop import ReActLoop

recorder = TraceRecorder()
loop = ReActLoop(llm, tools, recorder=recorder)   # 非侵入式，无 recorder 时行为不变
await loop.run("HCS 是什么？")
trace = recorder.traces[-1]   # 完整轨迹，可喂给任意指标计算器
```

---

## 五大指标详解

### 1. 任务成功率（看结果）

- `task_success_rate(traces, expected)` → 是否产出 Final Answer + 答案命中期望关键词。
- 离线：golden set 每例带 `expected_keywords`；线上：滚动窗口 `success_rate`。

### 2. 回答质量（看输出）

- `answer_quality(trace, reference)` → accuracy（参考答案重叠）/ completeness / conciseness / **hallucination_safety**。
- 幻觉检测：回答中的内容词有多少能在 `retrieved_context`（RAG 检索结果）里找到支撑。
- 可选 `prompts/eval_llm_judge_v1.txt` 用 LLM-as-judge 替代启发式打分。

### 3. 执行轨迹（看过程）

- `trajectory_quality(trace)` → 检测三种失败模式：
  - **死循环**：同一工具 + 同一参数重复调用
  - **死胡同**：达到 max_iterations 未产出 Final Answer
  - **乱结束**：no_action / error 终止
- 输出 0..1 分 + `issues` 明细（如 `dead_loop(2x)`、`max_iterations_dead_end`）。

### 4. 工具调用（看能力）

- `tool_call_quality(trace, expected_tools, tool_schemas)` →
  - **工具选择准确率**：应调用的工具是否被调用
  - **参数合法率**：args 是否满足 `required` + 类型声明（对照 MCP tool schema）
  - **调用错误率** + **冗余调用率**
- 综合分 = 0.4×选择 + 0.3×参数 + 0.2×(1-错误率) + 0.1×(1-冗余)。

### 5. 延迟和 Token（看工程落地）

- `aggregate_cost(traces)` → P50 / P95 延迟、平均 token、**每次成功任务的 token**（失败任务的 token 不算分母，反映真实效率）。
- P95 归一化到 0..1（≤2s=1.0，≥15s=0.0）参与综合分。

### 综合得分

```
overall = 0.25×任务成功率 + 0.20×回答质量 + 0.20×执行轨迹 + 0.20×工具调用 + 0.15×延迟归一
```

---

## 复用项目现有基础设施

| 现有模块 | 在评估框架中的角色 |
|----------|---------------------|
| `config/audit.py`（trace_id 全链路审计） | 线上评测的 trace 关联基础 |
| `agents/context_manager.py::count_tokens` | ReAct 插桩的 token 计数 |
| `mcp_server/protocol_handler.py`（tool schema + RBAC） | 工具调用指标的 schema 来源 |
| `agents/knowledge_qa/react_loop.py` | 执行轨迹 + 工具调用的采集点（已插桩） |
| `tests/test_agent_eval.py`（旧版 3 指标） | 本框架是其超集，旧指标仍保留 |

---

## 测试

```bash
.venv/Scripts/python.exe -m pytest tests/test_eval_framework.py -v   # 19 个测试，无需 API key
```
