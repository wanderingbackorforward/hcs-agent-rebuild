"""Report rendering — turns a metrics dict into a readable Markdown report.

Renders against the 5-metric template:

  1. 任务成功率  (task success rate)   — 看结果
  2. 回答质量    (answer quality)      — 看输出
  3. 执行轨迹    (execution trajectory) — 看过程
  4. 工具调用    (tool call)           — 看能力
  5. 延迟和Token (latency & token)     — 看工程落地
"""
from typing import Any, Dict, List

from eval.trace import AgentTrace


def render_report(
    report: Dict[str, Any],
    traces: List[AgentTrace],
    cases: List[Dict[str, Any]],
) -> str:
    """Render the full metrics report as Markdown."""
    lines: List[str] = []
    lines.append("# HCS Agent 评估报告")
    lines.append("")
    lines.append("> 5 核心指标 · 离线+线上双轨评测")
    lines.append("")
    lines.append("- 评测模式: **{}**".format(report.get("mode", "n/a")))
    lines.append("- 测试用例数: **{}**".format(report.get("case_count", 0)))
    lines.append("- 综合得分: **{:.3f}** / 1.000".format(report.get("overall_score", 0)))
    lines.append("")

    lines.append("## 五大核心指标总览")
    lines.append("")
    lines.append("| # | 指标 | 看什么 | 得分 |")
    lines.append("|---|------|--------|------|")
    _row(lines, "1", "任务成功率", "看结果", report, "1_task_success_rate", "task_success_rate")
    _row(lines, "2", "回答质量", "看输出", report, "2_answer_quality", _quality_head(report))
    _row(lines, "3", "执行轨迹", "看过程", report, "3_execution_trajectory", "trajectory_score")
    _row(lines, "4", "工具调用", "看能力", report, "4_tool_call", "tool_score")
    _row(lines, "5", "延迟和Token", "看工程落地", report, "5_latency_token", "p95_latency_ms", True)
    lines.append("")

    _section_success(lines, report)
    _section_quality(lines, report)
    _section_trajectory(lines, report)
    _section_tool(lines, report)
    _section_cost(lines, report)
    _section_per_case(lines, traces, cases)

    return "\n".join(lines)


def _row(lines, idx, name, view, report, key, score_key, is_raw=False):
    block = report.get(key, {})
    val = block.get(score_key, 0)
    if is_raw:
        shown = "{:.0f} ms".format(val)
    else:
        shown = "{:.3f}".format(val)
    lines.append("| {} | {} | {} | {} |".format(idx, name, view, shown))


def _quality_head(report) -> str:
    q = report.get("2_answer_quality", {})
    return "accuracy"  # representative; detail shown in section


def _section_success(lines, report):
    s = report.get("1_task_success_rate", {})
    lines.append("### 1. 任务成功率（看结果）")
    lines.append("")
    lines.append("- 任务成功率: **{:.1%}**".format(s.get("task_success_rate", 0)))
    lines.append("- 正常完成率: **{:.1%}**".format(s.get("finished_rate", 0)))
    lines.append("- 关键词命中率: **{:.1%}**".format(s.get("matched_rate", 0)))
    lines.append("")


def _section_quality(lines, report):
    q = report.get("2_answer_quality", {})
    lines.append("### 2. 回答质量（看输出）")
    lines.append("")
    lines.append("| 子维度 | 得分 |")
    lines.append("|--------|------|")
    for k in ("accuracy", "completeness", "conciseness", "hallucination_safety"):
        lines.append("| {} | {:.3f} |".format(k, q.get(k, 0)))
    lines.append("")
    lines.append("> 幻觉安全度(hallucination_safety)越高越好：1.0=回答可被检索上下文支撑。")
    lines.append("")


def _section_trajectory(lines, report):
    t = report.get("3_execution_trajectory", {})
    lines.append("### 3. 执行轨迹（看过程）")
    lines.append("")
    lines.append("- 轨迹得分: **{:.3f}**".format(t.get("trajectory_score", 0)))
    lines.append("- 平均步数: **{:.1f}**".format(t.get("avg_steps", 0)))
    lines.append("- 死循环发生率: **{:.1%}**".format(t.get("loop_rate", 0)))
    lines.append("- 干净终止率(到达Final Answer): **{:.1%}**".format(
        t.get("clean_termination_rate", 0)))
    lines.append("")
    lines.append("> 检测项：死循环(重复调用)、死胡同(达max迭代)、乱结束(无Final Answer)。")
    lines.append("")


def _section_tool(lines, report):
    t = report.get("4_tool_call", {})
    lines.append("### 4. 工具调用（看能力）")
    lines.append("")
    lines.append("| 子维度 | 得分 |")
    lines.append("|--------|------|")
    lines.append("| 综合得分 | {:.3f} |".format(t.get("tool_score", 0)))
    lines.append("| 工具选择准确率 | {:.3f} |".format(t.get("selection_accuracy", 0)))
    lines.append("| 参数合法率 | {:.3f} |".format(t.get("param_validity", 0)))
    lines.append("| 调用错误率 | {:.3f} |".format(t.get("error_rate", 0)))
    lines.append("| 平均调用次数 | {:.1f} |".format(t.get("avg_call_count", 0)))
    lines.append("| 冗余调用率 | {:.1%} |".format(t.get("redundant_rate", 0)))
    lines.append("")


def _section_cost(lines, report):
    c = report.get("5_latency_token", {})
    lines.append("### 5. 延迟和 Token（看工程落地）")
    lines.append("")
    lines.append("- P50 延迟: **{:.0f} ms**".format(c.get("p50_latency_ms", 0)))
    lines.append("- P95 延迟: **{:.0f} ms**".format(c.get("p95_latency_ms", 0)))
    lines.append("- 平均 Token: **{:.0f}**".format(c.get("avg_total_tokens", 0)))
    lines.append("- 每次成功任务的 Token: **{:.0f}**".format(c.get("tokens_per_success", 0)))
    lines.append("- 成功率: **{:.1%}**".format(c.get("success_rate", 0)))
    lines.append("")


def _section_per_case(lines, traces, cases):
    lines.append("### 逐用例明细")
    lines.append("")
    lines.append("| 用例 | 意图 | 步数 | 终止 | 成功 | 延迟ms | Token |")
    lines.append("|------|------|------|------|------|--------|-------|")
    for tr, case in zip(traces, cases):
        lines.append("| {} | {} | {} | {} | {} | {:.0f} | {} |".format(
            case.get("id", "-"), case.get("expected_intent", "-"),
            tr.step_count, tr.termination_reason or "-",
            "✓" if tr.success else "✗",
            tr.latency_ms, tr.total_tokens,
        ))
    lines.append("")
