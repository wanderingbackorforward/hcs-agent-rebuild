"""Intent routing accuracy evaluator.

Runs the golden test set through the TaskClassifier and computes:
  - Overall accuracy (correct routes / total cases)
  - Per-intent recall (correct / total per expected intent)
  - Confusion matrix (expected x predicted)
  - Detailed error list for debugging

Usage (live mode, requires LLM_API_KEY):

    from config.model_provider import create_chat_model
    from agents.task_classification.task_classifier import TaskClassifier
    from eval.intent_routing_eval import evaluate_intent_routing, format_report

    classifier = TaskClassifier(create_chat_model(temperature=0))
    result = asyncio.run(evaluate_intent_routing(classifier))
    print(format_report(result))

Usage (replay mode, no API key needed — for CI metrics logic tests):

    from eval.intent_routing_eval import evaluate_from_predictions, format_report

    result = evaluate_from_predictions(golden_cases, predictions)
    print(format_report(result))
"""
import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Canonical intent labels.
# To add a new intent:
#   1. Add it here and to AGENT_DESCRIPTIONS in nli_validator.py
#   2. Add golden cases below (recall + boundary confusion)
#   3. Add a router branch in agent_router.py and a display name in decision_explainer.py
#   4. Run full golden set — old intent accuracy must not drop (regression gate)
INTENTS = ("environment_match", "knowledge_qa", "unrelated")


# --------------------------------------------------------------------------- #
# Golden test cases — 55 labeled queries for intent routing evaluation.
# --------------------------------------------------------------------------- #

GOLDEN_CASES: List[Tuple[str, str]] = [
    # --- environment_match (25) ---
    ("帮我找一个 test 环境，要有 MySQL 和 Redis", "environment_match"),
    ("推荐一个 beijing 的 staging 环境", "environment_match"),
    ("我想确认当前环境是否可用", "environment_match"),
    ("查一下上海有哪些测试环境", "environment_match"),
    ("需要 Kafka + MongoDB 的环境", "environment_match"),
    ("给我找一个 dev 环境跑用例", "environment_match"),
    ("环境 hcs-test-01 的 MySQL 端口通吗", "environment_match"),
    ("哪个环境有 elasticsearch", "environment_match"),
    ("我要验证测试环境组件是否齐全", "environment_match"),
    ("匹配一个可用的测试环境", "environment_match"),
    ("找资源充足的环境", "environment_match"),
    ("列出 beijing 所有 available 的环境", "environment_match"),
    ("我需要 MySQL 8.0 的测试环境", "environment_match"),
    ("Redis 专项测试环境在哪里", "environment_match"),
    ("确认 staging 环境状态", "environment_match"),
    ("探测 10.0.1.10 的 3306 端口", "environment_match"),
    ("环境里有没有 kafka", "environment_match"),
    ("hcs-staging-01 能不能用", "environment_match"),
    ("给我推荐一个适合跑回归测试的环境", "environment_match"),
    ("测试环境需要包含 mongodb", "environment_match"),
    ("查看 hcs-test-02 的组件", "environment_match"),
    ("找一个 region 是上海的环境", "environment_match"),
    ("环境匹配：dev + mysql + redis", "environment_match"),
    ("确认所有组件 available", "environment_match"),
    ("哪个环境有 busy 状态", "environment_match"),
    # --- knowledge_qa (25) ---
    ("HCS SDK 怎么安装", "knowledge_qa"),
    ("如何初始化 HCS 客户端", "knowledge_qa"),
    ("测试环境规范要求是什么", "knowledge_qa"),
    ("HCS 部署分几个阶段", "knowledge_qa"),
    ("smoke test 是什么", "knowledge_qa"),
    ("Access Key 怎么配置", "knowledge_qa"),
    ("HCS 用户手册在哪里看", "knowledge_qa"),
    ("回归测试要做哪些事", "knowledge_qa"),
    ("HCS SDK 支持哪个 Python 版本", "knowledge_qa"),
    ("region 参数怎么填", "knowledge_qa"),
    ("MySQL 5.7 和 8.0 有什么区别", "knowledge_qa"),
    ("hcs-deploy 工具怎么用", "knowledge_qa"),
    ("验收阶段需要做什么", "knowledge_qa"),
    ("内部测试规范有什么要求", "knowledge_qa"),
    ("安装阶段要注意什么", "knowledge_qa"),
    ("准备阶段需要什么", "knowledge_qa"),
    ("HCS 混合云平台是什么", "knowledge_qa"),
    ("pip install hcs-sdk 之后怎么配置", "knowledge_qa"),
    ("网络规划在部署的哪个阶段", "knowledge_qa"),
    ("许可证是部署必须的吗", "knowledge_qa"),
    ("Redis 5.0+ 是强制要求吗", "knowledge_qa"),
    ("Kafka 版本要求是多少", "knowledge_qa"),
    ("组件必须处于 available 状态吗", "knowledge_qa"),
    ("端口不可连通怎么办", "knowledge_qa"),
    ("SDK 文档在哪里", "knowledge_qa"),
    # --- unrelated (5) ---
    ("今天天气怎么样", "unrelated"),
    ("帮我点外卖", "unrelated"),
    ("讲个笑话", "unrelated"),
    ("推荐一部电影", "unrelated"),
    ("你会写 Python 吗", "unrelated"),
]


# --------------------------------------------------------------------------- #
# Data models
# --------------------------------------------------------------------------- #

@dataclass
class IntentMetric:
    """Per-intent statistics."""
    intent: str
    total: int = 0
    correct: int = 0
    predicted: int = 0  # total times this intent was predicted (column sum)
    recall: float = 0.0   # correct / total (TP / (TP + FN))
    precision: float = 0.0  # correct / predicted (TP / (TP + FP))
    f1: float = 0.0      # 2 * P * R / (P + R)

    def compute_recall(self):
        self.recall = self.correct / self.total if self.total > 0 else 0.0

    def compute_precision(self):
        self.precision = self.correct / self.predicted if self.predicted > 0 else 0.0

    def compute_f1(self):
        if self.precision + self.recall > 0:
            self.f1 = 2 * self.precision * self.recall / (self.precision + self.recall)
        else:
            self.f1 = 0.0


@dataclass
class CaseResult:
    """Single golden case evaluation result."""
    query: str
    expected: str
    predicted: str
    correct: bool
    confidence: float = 1.0


@dataclass
class RoutingEvalResult:
    """Full evaluation result for the golden set."""
    total: int = 0
    correct: int = 0
    accuracy: float = 0.0
    per_intent: Dict[str, IntentMetric] = field(default_factory=dict)
    confusion_matrix: Dict[str, Dict[str, int]] = field(default_factory=dict)
    errors: List[CaseResult] = field(default_factory=list)
    all_results: List[CaseResult] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Evaluation logic
# --------------------------------------------------------------------------- #

def _normalize_intent(intent: str) -> str:
    """Normalize intent labels (handle environment_match vs environment_matching)."""
    if intent in ("environment_matching", "environment"):
        return "environment_match"
    return intent


def _init_metrics(cases: List[Tuple[str, str]]) -> Dict[str, IntentMetric]:
    metrics: Dict[str, IntentMetric] = {}
    for _, expected in cases:
        label = _normalize_intent(expected)
        if label not in metrics:
            metrics[label] = IntentMetric(intent=label)
        metrics[label].total += 1
    return metrics


def _init_confusion_matrix(labels: List[str]) -> Dict[str, Dict[str, int]]:
    matrix: Dict[str, Dict[str, int]] = {}
    for row in labels:
        matrix[row] = {col: 0 for col in labels}
    return matrix


def evaluate_from_predictions(
    cases: List[Tuple[str, str]],
    predictions: List[Tuple[str, str, float]],
) -> RoutingEvalResult:
    """Evaluate routing accuracy from pre-computed predictions.

    Args:
        cases: List of (query, expected_intent).
        predictions: List of (query, predicted_intent, confidence).
                     Must be same length and order as cases.

    Returns:
        RoutingEvalResult with all metrics computed.
    """
    assert len(cases) == len(predictions), (
        f"cases ({len(cases)}) vs predictions ({len(predictions)}) length mismatch"
    )

    labels = sorted(set(
        _normalize_intent(e) for _, e in cases
    ) | set(
        _normalize_intent(p) for _, p, _ in predictions
    ))
    per_intent = _init_metrics(cases)
    confusion = _init_confusion_matrix(labels)

    result = RoutingEvalResult()
    result.total = len(cases)

    for (query, expected), (pred_query, predicted, confidence) in zip(cases, predictions):
        exp_norm = _normalize_intent(expected)
        pred_norm = _normalize_intent(predicted)
        is_correct = exp_norm == pred_norm

        # Update confusion matrix (both row and col guaranteed to exist).
        confusion.setdefault(exp_norm, {l: 0 for l in labels})
        confusion[exp_norm].setdefault(pred_norm, 0)
        confusion[exp_norm][pred_norm] += 1

        # Update per-intent stats.
        if exp_norm in per_intent:
            if is_correct:
                per_intent[exp_norm].correct += 1

        case_result = CaseResult(
            query=query,
            expected=exp_norm,
            predicted=pred_norm,
            correct=is_correct,
            confidence=confidence,
        )
        result.all_results.append(case_result)
        if not is_correct:
            result.errors.append(case_result)

    result.correct = result.total - len(result.errors)
    result.accuracy = result.correct / result.total if result.total > 0 else 0.0

    # Compute predicted counts (column sums of confusion matrix) for precision.
    for intent in per_intent:
        per_intent[intent].predicted = sum(
            confusion[row].get(intent, 0) for row in confusion
        )

    for m in per_intent.values():
        m.compute_recall()
        m.compute_precision()
        m.compute_f1()
    result.per_intent = per_intent
    result.confusion_matrix = confusion

    return result


async def evaluate_intent_routing(classifier) -> RoutingEvalResult:
    """Run all golden cases through the classifier and compute metrics.

    Args:
        classifier: TaskClassifier instance (or any object with async classify()).

    Returns:
        RoutingEvalResult with all metrics computed.
    """
    predictions: List[Tuple[str, str, float]] = []

    for query, expected in GOLDEN_CASES:
        try:
            result = await classifier.classify(query)
            predicted = result.get("intent_type", "knowledge_qa")
            confidence = float(result.get("confidence", 1.0))
        except Exception as e:
            logger.warning("Classification failed for '%s': %s", query[:30], e)
            predicted = "unknown"
            confidence = 0.0
        predictions.append((query, predicted, confidence))

    return evaluate_from_predictions(GOLDEN_CASES, predictions)


# --------------------------------------------------------------------------- #
# Report formatting
# --------------------------------------------------------------------------- #

def format_report(result: RoutingEvalResult) -> str:
    """Generate a Markdown report from the evaluation result."""
    lines = [
        "# 意图路由准确率评估报告",
        "",
        "## 总体指标",
        "",
        f"| 指标 | 值 |",
        f"|------|------|",
        f"| 总用例数 | {result.total} |",
        f"| 正确路由数 | {result.correct} |",
        f"| 错误路由数 | {len(result.errors)} |",
        f"| **准确率** | **{result.accuracy:.1%}** |",
        "",
        "> 注：离线评估通常比线上高 5-10%（线上分布更复杂、有噪声输入）。",
        "",
        "## 各意图 Precision / Recall / F1",
        "",
        f"| 意图 | 总数 | 正确 | 预测数 | Precision | Recall | F1 |",
        f"|------|------|------|--------|-----------|--------|-----|",
    ]
    for intent in sorted(result.per_intent.keys()):
        m = result.per_intent[intent]
        lines.append(
            f"| {intent} | {m.total} | {m.correct} | {m.predicted} | "
            f"{m.precision:.1%} | {m.recall:.1%} | {m.f1:.3f} |"
        )

    lines.extend([
        "",
        "## 混淆矩阵 (行=期望, 列=预测)",
        "",
    ])
    labels = sorted(result.confusion_matrix.keys())
    header = "| 期望 \\ 预测 | " + " | ".join(labels) + " |"
    separator = "|---" * (len(labels) + 1) + "|"
    lines.append(header)
    lines.append(separator)
    for row_label in labels:
        row = result.confusion_matrix[row_label]
        cells = " | ".join(str(row.get(col, 0)) for col in labels)
        lines.append(f"| {row_label} | {cells} |")

    if result.errors:
        lines.extend([
            "",
            "## 错误案例分析",
            "",
            f"| # | 查询 | 期望意图 | 实际意图 | 置信度 |",
            f"|---|------|---------|---------|--------|",
        ])
        for i, err in enumerate(result.errors, 1):
            query_short = err.query[:40] + ("..." if len(err.query) > 40 else "")
            lines.append(
                f"| {i} | {query_short} | {err.expected} | {err.predicted} | {err.confidence:.2f} |"
            )

    return "\n".join(lines)


def print_report(result: RoutingEvalResult):
    """Print the evaluation report to stdout."""
    print(format_report(result))
