"""Golden tests for task classification intent routing.

Test layers:
  1. JSON parsing — parse_classification_json handles all intent labels
  2. FakeLLM end-to-end — classifier + JSON pipeline works with a stub LLM
  3. Metrics logic (replay) — accuracy / recall / confusion matrix from
     pre-computed predictions, no API key needed (CI-safe)
  4. Live LLM accuracy — runs all 55 golden cases through the real LLM,
     computes overall accuracy and per-intent recall, asserts >= 90%
"""
import json
import os
from types import SimpleNamespace

import pytest

from agents.task_classification.task_classifier import TaskClassifier
from agents.task_classification.json_utils import parse_classification_json
from eval.intent_routing_eval import (
    GOLDEN_CASES,
    evaluate_from_predictions,
    evaluate_intent_routing,
    format_report,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _has_api_key() -> bool:
    return bool(os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY"))


class FakeLLM:
    """A deterministic fake LLM for golden test validation."""

    def __init__(self, intent: str):
        self._intent = intent

    async def astream(self, messages):
        payload = json.dumps({
            "intent_type": self._intent,
            "confidence": 0.95,
            "required_fields": {},
            "missing_fields": [],
            "keywords": [],
            "topic": "test"
        })
        yield SimpleNamespace(content=payload)


# --------------------------------------------------------------------------- #
# Layer 1: JSON parsing
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("user_input, expected_intent", GOLDEN_CASES)
def test_parse_golden_json(user_input, expected_intent):
    """Verify classifier JSON parsing works for expected intent strings."""
    result = parse_classification_json(json.dumps({"intent_type": expected_intent}))
    assert result["intent_type"] == expected_intent


# --------------------------------------------------------------------------- #
# Layer 2: FakeLLM end-to-end pipeline
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
@pytest.mark.parametrize("user_input, expected_intent", GOLDEN_CASES[:3])
async def test_fake_llm_classification(user_input, expected_intent):
    """Classifier + JSON pipeline works with a deterministic stub LLM."""
    classifier = TaskClassifier(FakeLLM(expected_intent))
    result = await classifier.classify(user_input)
    assert result.get("intent_type") == expected_intent
    assert float(result.get("confidence", 0)) >= 0.9


# --------------------------------------------------------------------------- #
# Layer 3: Metrics logic (replay mode — no API key, CI-safe)
# --------------------------------------------------------------------------- #

class TestRoutingMetricsReplay:
    """Test accuracy / recall / confusion matrix calculation from
    pre-computed predictions. No LLM needed — pure logic tests."""

    def test_perfect_predictions(self):
        """100% accuracy when all predictions match expectations."""
        predictions = [(q, exp, 0.95) for q, exp in GOLDEN_CASES]
        result = evaluate_from_predictions(GOLDEN_CASES, predictions)

        assert result.total == len(GOLDEN_CASES)
        assert result.correct == len(GOLDEN_CASES)
        assert result.accuracy == 1.0
        assert len(result.errors) == 0
        for intent, metric in result.per_intent.items():
            assert metric.recall == 1.0

    def test_all_wrong_predictions(self):
        """0% accuracy when no predictions match."""
        wrong_map = {
            "environment_match": "knowledge_qa",
            "knowledge_qa": "unrelated",
            "unrelated": "environment_match",
        }
        predictions = [
            (q, wrong_map.get(exp, "unrelated"), 0.5)
            for q, exp in GOLDEN_CASES
        ]
        result = evaluate_from_predictions(GOLDEN_CASES, predictions)

        assert result.accuracy == 0.0
        assert len(result.errors) == len(GOLDEN_CASES)
        for metric in result.per_intent.values():
            assert metric.recall == 0.0

    def test_partial_accuracy_and_recall(self):
        """Verify recall calculation when only some intents are correct."""
        cases = [
            ("查环境A", "environment_match"),
            ("查环境B", "environment_match"),
            ("查文档A", "knowledge_qa"),
            ("查文档B", "knowledge_qa"),
            ("闲聊A", "unrelated"),
        ]
        # env all correct, kb half correct, unrelated wrong
        predictions = [
            ("查环境A", "environment_match", 0.9),
            ("查环境B", "environment_match", 0.9),
            ("查文档A", "knowledge_qa", 0.9),
            ("查文档B", "unrelated", 0.4),
            ("闲聊A", "knowledge_qa", 0.3),
        ]
        result = evaluate_from_predictions(cases, predictions)

        assert result.total == 5
        assert result.correct == 3
        assert result.accuracy == pytest.approx(0.6)
        assert result.per_intent["environment_match"].recall == 1.0
        assert result.per_intent["knowledge_qa"].recall == 0.5
        assert result.per_intent["unrelated"].recall == 0.0

    def test_confusion_matrix_structure(self):
        """Confusion matrix has correct dimensions and values."""
        cases = [
            ("q1", "environment_match"),
            ("q2", "knowledge_qa"),
        ]
        predictions = [
            ("q1", "environment_match", 0.9),
            ("q2", "unrelated", 0.3),
        ]
        result = evaluate_from_predictions(cases, predictions)

        labels = sorted(result.confusion_matrix.keys())
        assert "environment_match" in labels
        assert "knowledge_qa" in labels
        # q1: env -> env (diagonal)
        assert result.confusion_matrix["environment_match"]["environment_match"] == 1
        # q2: kb -> unrelated (off-diagonal)
        assert result.confusion_matrix["knowledge_qa"]["unrelated"] == 1

    def test_error_list_contents(self):
        """Error list contains only misclassified cases with details."""
        cases = [
            ("查环境", "environment_match"),
            ("查文档", "knowledge_qa"),
        ]
        predictions = [
            ("查环境", "environment_match", 0.9),
            ("查文档", "unrelated", 0.3),
        ]
        result = evaluate_from_predictions(cases, predictions)

        assert len(result.errors) == 1
        err = result.errors[0]
        assert err.query == "查文档"
        assert err.expected == "knowledge_qa"
        assert err.predicted == "unrelated"
        assert err.correct is False

    def test_report_generation(self):
        """format_report produces valid Markdown with all sections."""
        predictions = [(q, exp, 0.9) for q, exp in GOLDEN_CASES[:5]]
        result = evaluate_from_predictions(GOLDEN_CASES[:5], predictions)
        report = format_report(result)

        assert "意图路由准确率评估报告" in report
        assert "总体指标" in report
        assert "各意图召回率" in report
        assert "混淆矩阵" in report
        assert "100.0%" in report

    def test_full_golden_set_count(self):
        """Golden case set has exactly 55 cases covering 3 intents."""
        assert len(GOLDEN_CASES) == 55
        intents = set(exp for _, exp in GOLDEN_CASES)
        assert intents == {"environment_match", "knowledge_qa", "unrelated"}
        # Distribution check
        counts = {}
        for _, exp in GOLDEN_CASES:
            counts[exp] = counts.get(exp, 0) + 1
        assert counts["environment_match"] == 25
        assert counts["knowledge_qa"] == 25
        assert counts["unrelated"] == 5


# --------------------------------------------------------------------------- #
# Layer 4: Live LLM accuracy (requires API key)
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(not _has_api_key(), reason="LLM_API_KEY not set")
class TestLiveLLMAccuracy:
    """Run all golden cases through the real LLM and compute metrics."""

    @pytest.mark.asyncio
    async def test_routing_accuracy(self):
        """Overall routing accuracy must be >= 90%."""
        from config.model_provider import create_chat_model

        classifier = TaskClassifier(create_chat_model(temperature=0))
        result = await evaluate_intent_routing(classifier)

        report = format_report(result)
        print("\n" + report)

        assert result.total == 55, f"Expected 55 cases, got {result.total}"
        assert result.accuracy >= 0.90, (
            f"Routing accuracy {result.accuracy:.1%} below 90% threshold.\n"
            f"Errors ({len(result.errors)}):\n"
            + "\n".join(
                f"  [{e.expected} -> {e.predicted}] {e.query}"
                for e in result.errors
            )
        )

    @pytest.mark.asyncio
    async def test_per_intent_recall(self):
        """Each intent must have recall >= 85%."""
        from config.model_provider import create_chat_model

        classifier = TaskClassifier(create_chat_model(temperature=0))
        result = await evaluate_intent_routing(classifier)

        for intent, metric in result.per_intent.items():
            assert metric.recall >= 0.85, (
                f"Recall for '{intent}' is {metric.recall:.1%} "
                f"(correct={metric.correct}/{metric.total}), below 85%"
            )
