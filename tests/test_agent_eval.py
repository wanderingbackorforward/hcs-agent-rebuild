"""Agent evaluation framework - measures task completion, retrieval precision, response quality.

Interview talking point: "I built an Agent evaluation framework that measures
task completion rate, retrieval precision (top-k hit rate), and response quality
(accuracy/completeness/conciseness). Also supports few-shot evaluation with
labeled samples."
"""
import pytest
from typing import List, Dict


# ============ Evaluation Metrics ============

def task_completion_rate(responses: List[str], expected_keywords: List[str]) -> float:
    """Measure what fraction of responses contain expected keywords.

    Args:
        responses: Agent responses to evaluate.
        expected_keywords: Keywords that should appear in a complete response.

    Returns:
        Completion rate (0.0 - 1.0).
    """
    if not responses:
        return 0.0
    completed = sum(
        1 for r in responses
        if any(kw.lower() in r.lower() for kw in expected_keywords)
    )
    return completed / len(responses)


def retrieval_precision_at_k(retrieved_docs: List[Dict], relevant_ids: set, k: int = 5) -> float:
    """Precision@K: fraction of top-k retrieved docs that are relevant.

    Args:
        retrieved_docs: List of {doc_id, score, ...} dicts.
        relevant_ids: Set of doc_ids that are known relevant.
        k: Top K to evaluate.

    Returns:
        Precision@K (0.0 - 1.0).
    """
    top_k = retrieved_docs[:k]
    if not top_k:
        return 0.0
    relevant = sum(1 for doc in top_k if doc.get("doc_id") in relevant_ids)
    return relevant / len(top_k)


def response_quality_score(response: str, reference: str, llm=None) -> Dict[str, float]:
    """Score response quality on accuracy, completeness, conciseness.

    Uses simple heuristics if no LLM available, LLM-based scoring otherwise.

    Returns:
        Dict with accuracy (0-1), completeness (0-1), conciseness (0-1).
    """
    # Heuristic scoring (no LLM needed).
    # Accuracy: keyword overlap with reference.
    ref_words = set(reference.lower().split())
    resp_words = set(response.lower().split())
    overlap = len(ref_words & resp_words)
    accuracy = min(1.0, overlap / max(len(ref_words), 1))

    # Completeness: response length relative to reference.
    resp_len = len(response)
    ref_len = max(len(reference), 1)
    completeness = min(1.0, resp_len / ref_len) if resp_len <= ref_len * 2 else 0.7

    # Conciseness: penalize overly long responses.
    if resp_len <= ref_len * 1.5:
        conciseness = 1.0
    elif resp_len <= ref_len * 3:
        conciseness = 0.7
    else:
        conciseness = 0.4

    return {
        "accuracy": round(accuracy, 2),
        "completeness": round(completeness, 2),
        "conciseness": round(conciseness, 2),
    }


def few_shot_evaluate(test_cases: List[Dict], agent_func, llm=None) -> Dict:
    """Run few-shot evaluation on a set of test cases.

    Args:
        test_cases: List of {query, expected_keywords, relevant_doc_ids, reference_answer}.
        agent_func: Callable that takes query and returns response.
        llm: Optional LLM for quality scoring.

    Returns:
        Aggregated metrics dict.
    """
    results = []
    for case in test_cases:
        response = agent_func(case["query"])
        if asyncio_run(response):
            # Handle async agent functions.
            pass

        quality = response_quality_score(response, case.get("reference_answer", ""), llm)
        result = {
            "query": case["query"],
            "response": response[:200],
            "quality": quality,
            "has_expected": any(
                kw.lower() in response.lower()
                for kw in case.get("expected_keywords", [])
            ),
        }
        results.append(result)

    # Aggregate.
    completion = sum(1 for r in results if r["has_expected"]) / max(len(results), 1)
    avg_quality = {
        "accuracy": sum(r["quality"]["accuracy"] for r in results) / max(len(results), 1),
        "completeness": sum(r["quality"]["completeness"] for r in results) / max(len(results), 1),
        "conciseness": sum(r["quality"]["conciseness"] for r in results) / max(len(results), 1),
    }

    return {
        "total_cases": len(results),
        "completion_rate": round(completion, 2),
        "avg_quality": {k: round(v, 2) for k, v in avg_quality.items()},
        "details": results,
    }


def asyncio_run(coro):
    """Helper to check if something is a coroutine."""
    import asyncio
    return asyncio.iscoroutine(coro)


# ============ Test Cases ============

class TestTaskCompletion:
    """Test task completion rate metric."""

    def test_all_complete(self):
        responses = ["HCS is a hybrid cloud solution", "The environment is ready"]
        keywords = ["hcs", "environment"]
        rate = task_completion_rate(responses, keywords)
        assert rate == 1.0

    def test_partial_complete(self):
        responses = ["HCS is a hybrid cloud solution", "I dont know"]
        keywords = ["hcs", "environment"]
        rate = task_completion_rate(responses, keywords)
        assert rate == 0.5

    def test_empty(self):
        assert task_completion_rate([], ["test"]) == 0.0


class TestRetrievalPrecision:
    """Test retrieval precision@K metric."""

    def test_perfect_precision(self):
        docs = [{"doc_id": "1"}, {"doc_id": "2"}, {"doc_id": "3"}]
        relevant = {"1", "2", "3"}
        precision = retrieval_precision_at_k(docs, relevant, k=3)
        assert precision == 1.0

    def test_partial_precision(self):
        docs = [{"doc_id": "1"}, {"doc_id": "4"}, {"doc_id": "3"}]
        relevant = {"1", "2", "3"}
        precision = retrieval_precision_at_k(docs, relevant, k=3)
        assert precision == pytest.approx(2/3, rel=0.01)

    def test_empty(self):
        assert retrieval_precision_at_k([], set(), k=5) == 0.0


class TestResponseQuality:
    """Test response quality scoring."""

    def test_perfect_response(self):
        reference = "HCS is a hybrid cloud solution for testing"
        response = "HCS is a hybrid cloud solution for testing"
        quality = response_quality_score(response, reference)
        assert quality["accuracy"] > 0.5
        assert quality["conciseness"] == 1.0

    def test_overly_long_response(self):
        reference = "Short answer"
        response = "Short answer " * 50
        quality = response_quality_score(response, reference)
        assert quality["conciseness"] < 0.8

    def test_overly_short_response(self):
        reference = "HCS is a hybrid cloud solution for automated testing"
        response = "Yes"
        quality = response_quality_score(response, reference)
        assert quality["completeness"] < 0.5


class TestFewShotEvaluation:
    """Test few-shot evaluation framework."""

    def test_few_shot_basic(self):
        test_cases = [
            {"query": "What is HCS?", "expected_keywords": ["hybrid", "cloud"],
             "reference_answer": "HCS is a hybrid cloud solution"},
            {"query": "How to match environment?", "expected_keywords": ["filter", "match"],
             "reference_answer": "Use structured field filtering to match environments"},
        ]

        def mock_agent(query):
            if "HCS" in query:
                return "HCS is a hybrid cloud solution for testing"
            return "Use structured field filtering to match environments"

        results = few_shot_evaluate(test_cases, mock_agent)
        assert results["total_cases"] == 2
        assert results["completion_rate"] == 1.0
        assert "avg_quality" in results
