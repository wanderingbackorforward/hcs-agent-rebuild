"""Tests for RAG evaluation framework: cases, RAGAS metrics, regression gate.

Test coverage:
  1. RAG test set: case count, categories, structure.
  2. RAGAS metrics: mock LLM, verify metric computation logic.
  3. Regression gate: baseline save/load, diff comparison, threshold gating.
  4. RAG evaluator mock mode: end-to-end with synthetic data.
  5. StageTimer: timing context manager.
"""
import json
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from eval.rag_cases import get_rag_cases, get_cases_by_category, get_category_summary
from eval.ragas_metrics import (
    RAGASSample,
    RAGASResult,
    evaluate_sample,
    evaluate_batch,
    sample_from_trace,
)
from eval.regression_gate import RegressionGate, RegressionReport, MetricDiff
from eval.rag_evaluator import RAGEvaluator, RAGEvalReport, RAGCaseResult
from eval.trace import StageTimer, TraceRecorder


# ---------------------------------------------------------------------------
# 1. RAG test set
# ---------------------------------------------------------------------------

class TestRAGCases:
    def test_case_count(self):
        cases = get_rag_cases()
        assert len(cases) >= 15, f"Expected >= 15 cases, got {len(cases)}"

    def test_all_cases_have_required_fields(self):
        for case in get_rag_cases():
            assert "id" in case
            assert "query" in case
            assert "reference_answer" in case
            assert "category" in case
            assert "expected_context_keywords" in case
            assert "min_context_relevance" in case

    def test_category_diversity(self):
        summary = get_category_summary()
        # Should cover at least 5 different categories
        assert len(summary) >= 5, f"Only {len(summary)} categories: {summary}"
        # Should have factual_qa
        assert "factual_qa" in summary
        # Should have multi_hop
        assert "multi_hop" in summary
        # Should have no_result or out_of_scope
        assert "no_result" in summary or "out_of_scope" in summary

    def test_unique_ids(self):
        cases = get_rag_cases()
        ids = [c["id"] for c in cases]
        assert len(ids) == len(set(ids)), "Duplicate case IDs found"

    def test_filter_by_category(self):
        factual = get_cases_by_category("factual_qa")
        assert len(factual) >= 3
        for case in factual:
            assert case["category"] == "factual_qa"


# ---------------------------------------------------------------------------
# 2. RAGAS metrics (with mocked LLM)
# ---------------------------------------------------------------------------

class TestRAGASMetrics:
    @pytest.fixture
    def sample_perfect(self):
        return RAGASSample(
            question="What is HCS?",
            answer="HCS is a hybrid cloud solution for testing.",
            contexts=["HCS is a hybrid cloud solution used for testing environment management."],
            reference="HCS is a hybrid cloud solution for testing.",
        )

    @pytest.fixture
    def sample_empty(self):
        return RAGASSample(
            question="What is HCS?",
            answer="",
            contexts=["Some context"],
            reference="HCS is a hybrid cloud solution.",
        )

    def test_empty_answer_returns_zeros(self, sample_empty):
        with patch("eval.ragas_metrics._llm_invoke", return_value=""):
            result = evaluate_sample(sample_empty)
        assert result.faithfulness == 0.0
        assert result.answer_relevance == 0.0

    def test_empty_contexts_returns_zero_faithfulness(self):
        sample = RAGASSample(
            question="What is HCS?",
            answer="HCS is a hybrid cloud.",
            contexts=[],
            reference="HCS is a hybrid cloud.",
        )
        with patch("eval.ragas_metrics._llm_invoke", return_value=""):
            result = evaluate_sample(sample)
        assert result.faithfulness == 0.0

    def test_faithfulness_with_mocked_claims(self, sample_perfect):
        """Mock LLM to return specific claims and verification."""
        # Mock: first call extracts claims, second verifies them
        mock_responses = [
            '["HCS is a hybrid cloud solution for testing"]',  # claims
            '[{"claim": "HCS is a hybrid cloud solution for testing", "supported": true}]',  # verify
            '{"score": 0.9, "reason": "directly answers"}',  # answer relevance
            '[{"chunk_index": 0, "relevant": true}]',  # context precision
            '{"supported_sentences": 1, "total_sentences": 1}',  # context recall
        ]
        with patch("eval.ragas_metrics._llm_invoke", side_effect=mock_responses):
            result = evaluate_sample(sample_perfect)

        assert result.faithfulness == 1.0
        assert result.answer_relevance == 0.9
        assert result.context_precision == 1.0
        assert result.context_recall == 1.0

    def test_faithfulness_partial_support(self):
        sample = RAGASSample(
            question="What is HCS?",
            answer="HCS is a hybrid cloud solution. It supports quantum computing.",
            contexts=["HCS is a hybrid cloud solution for testing."],
            reference="HCS is a hybrid cloud solution.",
        )
        mock_responses = [
            '["HCS is a hybrid cloud solution", "It supports quantum computing"]',
            '[{"claim": "HCS is a hybrid cloud solution", "supported": true}, '
            '{"claim": "It supports quantum computing", "supported": false}]',
            '{"score": 0.5}',
            '[{"chunk_index": 0, "relevant": true}]',
            '{"supported_sentences": 1, "total_sentences": 1}',
        ]
        with patch("eval.ragas_metrics._llm_invoke", side_effect=mock_responses):
            result = evaluate_sample(sample)

        # 1 out of 2 claims supported = 0.5
        assert result.faithfulness == 0.5

    def test_batch_evaluation(self):
        samples = [
            RAGASSample(
                question="Q1",
                answer="Answer 1",
                contexts=["Context 1"],
                reference="Reference 1",
            ),
            RAGASSample(
                question="Q2",
                answer="Answer 2",
                contexts=["Context 2"],
                reference="Reference 2",
            ),
        ]
        mock_responses = [
            # Sample 1
            '["Answer 1"]',
            '[{"claim": "Answer 1", "supported": true}]',
            '{"score": 0.8}',
            '[{"chunk_index": 0, "relevant": true}]',
            '{"supported_sentences": 1, "total_sentences": 1}',
            # Sample 2
            '["Answer 2"]',
            '[{"claim": "Answer 2", "supported": true}]',
            '{"score": 0.9}',
            '[{"chunk_index": 0, "relevant": true}]',
            '{"supported_sentences": 1, "total_sentences": 1}',
        ]
        with patch("eval.ragas_metrics._llm_invoke", side_effect=mock_responses):
            result = evaluate_batch(samples)

        assert result.sample_count == 2
        assert result.faithfulness == 1.0
        assert 0.8 <= result.answer_relevance <= 0.9

    def test_sample_from_trace(self):
        chunks = [
            {"content": "chunk 1", "score": 0.9},
            {"text": "chunk 2", "score": 0.7},
        ]
        sample = sample_from_trace(
            question="What is HCS?",
            answer="HCS is a cloud solution.",
            retrieved_chunks=chunks,
            reference="HCS is a hybrid cloud.",
        )
        assert len(sample.contexts) == 2
        assert sample.contexts[0] == "chunk 1"
        assert sample.contexts[1] == "chunk 2"

    def test_error_handling(self):
        """Metrics should return 0.0 on LLM errors, not crash."""
        sample = RAGASSample(
            question="Q",
            answer="A",
            contexts=["C"],
            reference="R",
        )
        with patch("eval.ragas_metrics._llm_invoke", side_effect=Exception("API error")):
            result = evaluate_sample(sample)

        assert result.faithfulness == 0.0
        assert len(result.errors) > 0


# ---------------------------------------------------------------------------
# 3. Regression gate
# ---------------------------------------------------------------------------

class TestRegressionGate:
    @pytest.fixture
    def temp_gate(self, tmp_path):
        return RegressionGate(baseline_dir=tmp_path)

    def test_first_run_passes(self, temp_gate):
        """First run with no baseline should pass."""
        current = {
            "git_commit": "abc123",
            "ragas": {"faithfulness": 0.8, "answer_relevance": 0.7,
                      "context_precision": 0.6, "context_recall": 0.5},
            "performance": {"avg_end_to_end_ms": 1000},
        }
        report, passed = temp_gate.evaluate_and_gate(current, save_on_pass=True)
        assert passed
        assert report.gate_result == "PASS"

    def test_save_and_load_baseline(self, temp_gate):
        current = {
            "git_commit": "abc123",
            "timestamp": "2026-01-01T00:00:00",
            "ragas": {"faithfulness": 0.8},
            "performance": {"avg_end_to_end_ms": 1000},
        }
        temp_gate.save_baseline(current)
        loaded = temp_gate.load_latest_baseline()
        assert loaded is not None
        assert loaded["git_commit"] == "abc123"
        assert loaded["ragas"]["faithfulness"] == 0.8

    def test_regression_detected(self, temp_gate):
        """Faithfulness drop > 5% should fail."""
        baseline = {
            "git_commit": "old",
            "timestamp": "2026-01-01",
            "ragas": {"faithfulness": 0.85, "answer_relevance": 0.8,
                      "context_precision": 0.7, "context_recall": 0.6},
            "performance": {"avg_end_to_end_ms": 1000},
        }
        temp_gate.save_baseline(baseline)

        current = {
            "git_commit": "new",
            "ragas": {"faithfulness": 0.70, "answer_relevance": 0.8,
                      "context_precision": 0.7, "context_recall": 0.6},
            "performance": {"avg_end_to_end_ms": 1000},
        }
        report, passed = temp_gate.evaluate_and_gate(current, save_on_pass=False)

        assert not passed
        assert "faithfulness" in report.failed_metrics

    def test_improvement_passes(self, temp_gate):
        """Metrics improvement should pass."""
        baseline = {
            "git_commit": "old",
            "timestamp": "2026-01-01",
            "ragas": {"faithfulness": 0.80, "answer_relevance": 0.80,
                      "context_precision": 0.80, "context_recall": 0.80},
            "performance": {"avg_end_to_end_ms": 1000},
        }
        temp_gate.save_baseline(baseline)

        current = {
            "git_commit": "new",
            "ragas": {"faithfulness": 0.85, "answer_relevance": 0.85,
                      "context_precision": 0.85, "context_recall": 0.85},
            "performance": {"avg_end_to_end_ms": 1000},
        }
        report, passed = temp_gate.evaluate_and_gate(current, save_on_pass=False)

        assert passed
        assert all(d.diff >= 0 for d in report.metric_diffs)

    def test_latency_regression_detected(self, temp_gate):
        """Latency increase > 20% should fail."""
        baseline = {
            "git_commit": "old",
            "timestamp": "2026-01-01",
            "ragas": {"faithfulness": 0.8, "answer_relevance": 0.8,
                      "context_precision": 0.8, "context_recall": 0.8},
            "performance": {"avg_end_to_end_ms": 1000, "avg_retrieval_ms": 200},
        }
        temp_gate.save_baseline(baseline)

        current = {
            "git_commit": "new",
            "ragas": {"faithfulness": 0.8, "answer_relevance": 0.8,
                      "context_precision": 0.8, "context_recall": 0.8},
            "performance": {"avg_end_to_end_ms": 1500, "avg_retrieval_ms": 200},
        }
        report, passed = temp_gate.evaluate_and_gate(current, save_on_pass=False)

        assert not passed
        assert "avg_end_to_end_ms" in report.failed_metrics

    def test_custom_thresholds(self, tmp_path):
        """Stricter threshold should catch smaller drops."""
        gate = RegressionGate(
            baseline_dir=tmp_path,
            thresholds={"faithfulness": 0.02},  # 2% threshold
        )
        baseline = {
            "git_commit": "old",
            "timestamp": "2026-01-01",
            "ragas": {"faithfulness": 0.80},
            "performance": {},
        }
        gate.save_baseline(baseline)

        current = {
            "git_commit": "new",
            "ragas": {"faithfulness": 0.76},  # 4% drop
            "performance": {},
        }
        report, passed = gate.evaluate_and_gate(current, save_on_pass=False)

        # 4% drop > 2% threshold → fail
        assert not passed

    def test_print_report(self, temp_gate, capsys):
        """print_report should output to stdout without error."""
        report = RegressionReport(
            gate_result="PASS",
            current_commit="abc",
            baseline_commit="def",
            summary="All good.",
        )
        temp_gate.print_report(report)
        captured = capsys.readouterr()
        assert "PASS" in captured.out


# ---------------------------------------------------------------------------
# 4. RAG evaluator mock mode
# ---------------------------------------------------------------------------

class TestRAGEvaluator:
    def test_mock_mode_run(self):
        """Mock mode should run end-to-end without LLM API."""
        evaluator = RAGEvaluator(mode="mock", skip_ragas=True)
        report = evaluator.run()

        assert report.mode == "mock"
        assert report.case_count == 15
        assert report.success_count == 15
        assert report.fail_count == 0
        assert report.avg_retrieval_ms > 0
        assert report.avg_rerank_ms > 0
        assert report.avg_generation_ms > 0
        assert report.p95_end_to_end_ms > 0

    def test_mock_mode_with_ragas_skipped(self):
        evaluator = RAGEvaluator(mode="mock", skip_ragas=True)
        report = evaluator.run()

        # RAGAS metrics should be 0 when skipped
        assert report.faithfulness == 0.0
        assert report.ragas_average == 0.0

    def test_report_json_serializable(self):
        evaluator = RAGEvaluator(mode="mock", skip_ragas=True)
        report = evaluator.run()

        json_str = report.to_json()
        data = json.loads(json_str)
        assert data["mode"] == "mock"
        assert data["case_count"] == 15
        assert "performance" in data
        assert "per_case" in data

    def test_custom_cases(self):
        custom = [
            {
                "id": "custom-1",
                "query": "test query",
                "reference_answer": "test answer",
                "category": "factual_qa",
                "expected_context_keywords": ["test"],
                "min_context_relevance": 0.5,
            },
        ]
        evaluator = RAGEvaluator(mode="mock", cases=custom, skip_ragas=True)
        report = evaluator.run()

        assert report.case_count == 1
        assert report.per_case[0]["case_id"] == "custom-1"


# ---------------------------------------------------------------------------
# 5. StageTimer
# ---------------------------------------------------------------------------

class TestStageTimer:
    def test_timing_without_recorder(self):
        """StageTimer works standalone without a TraceRecorder."""
        with StageTimer(stage="retrieval") as t:
            time.sleep(0.01)

        assert t.elapsed_ms > 5  # Should be at least ~10ms
        assert t.elapsed_ms < 100  # But not too long

    def test_timing_with_recorder(self):
        """StageTimer records timing on the TraceRecorder."""
        recorder = TraceRecorder()
        recorder.start("test query")

        with StageTimer(recorder, "retrieval"):
            time.sleep(0.01)

        with StageTimer(recorder, "generation"):
            time.sleep(0.01)

        trace = recorder.finalize(final_answer="test", success=True, termination_reason="done")
        assert "retrieval" in trace.stage_timings
        assert "generation" in trace.stage_timings
        assert trace.stage_timings["retrieval"] > 5
        assert trace.stage_timings["generation"] > 5

    def test_retrieved_chunks_setter(self):
        recorder = TraceRecorder()
        recorder.start("test")
        chunks = [{"content": "chunk1"}, {"content": "chunk2"}]
        recorder.set_retrieved_chunks(chunks)
        trace = recorder.finalize(final_answer="ans", success=True, termination_reason="done")
        assert len(trace.retrieved_chunks) == 2

    def test_trace_to_dict_includes_new_fields(self):
        recorder = TraceRecorder()
        recorder.start("test query")
        recorder.set_stage_timing("retrieval", 120.5)
        recorder.set_retrieved_chunks([{"content": "ctx"}])
        trace = recorder.finalize(final_answer="answer", success=True, termination_reason="done")

        d = trace.to_dict()
        assert "stage_timings" in d
        assert d["stage_timings"]["retrieval"] == 120.5
        assert "retrieved_chunks" in d
        assert len(d["retrieved_chunks"]) == 1
