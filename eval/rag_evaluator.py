"""RAG Evaluator — runs the full RAG pipeline on test cases and evaluates.

This is the core of the CI/CD regression gate. For each test case:
1. Calls the KnowledgeQAAgent (or directly HybridSearch) to run the full
   retrieval → rerank → generation pipeline.
2. Captures: generated answer, retrieved context chunks, per-stage latency.
3. Calls RAGAS-aligned metrics (ragas_metrics.evaluate_sample).
4. Aggregates results into a RAGEvalReport.

Usage in CI::

    from eval.rag_evaluator import RAGEvaluator
    evaluator = RAGEvaluator(mode="live")
    report = evaluator.run()
    # report.to_dict() → JSON-serializable for baseline storage

Modes:
  - "live": Calls the real KnowledgeQAAgent with the real HybridSearch.
            Full end-to-end test. Requires LLM API access.
  - "mock": Uses mock data for testing the eval framework itself.
            No LLM API needed. Useful for CI dry-runs.
"""
import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .rag_cases import get_rag_cases
from .ragas_metrics import (
    RAGASSample,
    RAGASBatchResult,
    evaluate_batch,
    evaluate_sample,
    sample_from_trace,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RAGCaseResult:
    """Result of running a single RAG test case."""
    case_id: str = ""
    query: str = ""
    category: str = ""
    answer: str = ""
    reference_answer: str = ""
    retrieved_chunks: List[Dict[str, Any]] = field(default_factory=list)
    stage_timings: Dict[str, float] = field(default_factory=dict)
    end_to_end_ms: float = 0.0
    success: bool = False
    error: str = ""
    ragas: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "query": self.query,
            "category": self.category,
            "answer": self.answer[:500],
            "reference_answer": self.reference_answer,
            "retrieved_chunk_count": len(self.retrieved_chunks),
            "stage_timings": self.stage_timings,
            "end_to_end_ms": round(self.end_to_end_ms, 1),
            "success": self.success,
            "error": self.error,
            "ragas": self.ragas,
        }


@dataclass
class RAGEvalReport:
    """Aggregated report from RAG evaluation run."""
    timestamp: str = ""
    git_commit: str = ""
    mode: str = "live"
    case_count: int = 0
    success_count: int = 0
    fail_count: int = 0

    # RAGAS metrics (averaged across all successful cases).
    faithfulness: float = 0.0
    answer_relevance: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0
    ragas_average: float = 0.0

    # Performance metrics (averaged across all successful cases).
    avg_retrieval_ms: float = 0.0
    avg_rerank_ms: float = 0.0
    avg_generation_ms: float = 0.0
    avg_end_to_end_ms: float = 0.0
    p95_end_to_end_ms: float = 0.0

    per_case: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "git_commit": self.git_commit,
            "mode": self.mode,
            "case_count": self.case_count,
            "success_count": self.success_count,
            "fail_count": self.fail_count,
            "ragas": {
                "faithfulness": round(self.faithfulness, 4),
                "answer_relevance": round(self.answer_relevance, 4),
                "context_precision": round(self.context_precision, 4),
                "context_recall": round(self.context_recall, 4),
                "average": round(self.ragas_average, 4),
            },
            "performance": {
                "avg_retrieval_ms": round(self.avg_retrieval_ms, 1),
                "avg_rerank_ms": round(self.avg_rerank_ms, 1),
                "avg_generation_ms": round(self.avg_generation_ms, 1),
                "avg_end_to_end_ms": round(self.avg_end_to_end_ms, 1),
                "p95_end_to_end_ms": round(self.p95_end_to_end_ms, 1),
            },
            "per_case": self.per_case,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class RAGEvaluator:
    """Runs RAG test cases through the full pipeline and evaluates with RAGAS.

    Args:
        mode: "live" (real pipeline) or "mock" (synthetic data for testing).
        cases: Custom case list. If None, uses get_rag_cases().
        skip_ragas: If True, skip LLM-as-judge evaluation (for perf-only runs).
    """

    def __init__(
        self,
        mode: str = "live",
        cases: Optional[List[Dict[str, Any]]] = None,
        skip_ragas: bool = False,
    ):
        self.mode = mode
        self.cases = cases or get_rag_cases()
        self.skip_ragas = skip_ragas

    def run(self) -> RAGEvalReport:
        """Run all cases and return aggregated report.

        This is the main entry point for CI. It's synchronous (wraps async
        internally) to be callable from a simple script.
        """
        report = RAGEvalReport(
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            mode=self.mode,
            case_count=len(self.cases),
        )
        report.git_commit = self._get_git_commit()

        results: List[RAGCaseResult] = []
        for i, case in enumerate(self.cases):
            logger.info("[%d/%d] Running case %s: %s",
                        i + 1, len(self.cases), case["id"], case["query"][:50])
            result = self._run_single_case(case)
            results.append(result)
            report.per_case.append(result.to_dict())

        self._aggregate(results, report)
        return report

    def _run_single_case(self, case: Dict[str, Any]) -> RAGCaseResult:
        """Run a single test case through the RAG pipeline."""
        result = RAGCaseResult(
            case_id=case["id"],
            query=case["query"],
            category=case["category"],
            reference_answer=case.get("reference_answer", ""),
        )

        try:
            if self.mode == "live":
                self._run_live(case, result)
            else:
                self._run_mock(case, result)
        except Exception as e:
            result.success = False
            result.error = str(e)
            logger.error("Case %s failed: %s", case["id"], e)

        # RAGAS evaluation (skip on failure or if disabled)
        if result.success and not self.skip_ragas:
            try:
                sample = RAGASSample(
                    question=result.query,
                    answer=result.answer,
                    contexts=[
                        chunk.get("content") or chunk.get("text") or ""
                        for chunk in result.retrieved_chunks
                    ],
                    reference=result.reference_answer,
                )
                ragas_result = evaluate_sample(sample)
                result.ragas = ragas_result.to_dict()
            except Exception as e:
                logger.error("RAGAS evaluation failed for %s: %s", case["id"], e)
                result.ragas = {"error": str(e)}

        return result

    def _run_live(self, case: Dict[str, Any], result: RAGCaseResult) -> None:
        """Run case through the real KnowledgeQAAgent pipeline."""
        from agents.knowledge_qa_agent import KnowledgeQAAgent

        agent = KnowledgeQAAgent()

        # Collect SSE events to extract answer and timings.
        start = time.time()
        answer_parts: List[str] = []
        stage_timings: Dict[str, float] = {}

        async def _run():
            async for event in agent.process_stream(case["query"]):
                if hasattr(event, "event_type"):
                    if event.event_type == "status":
                        # Status events tell us which stage we're in.
                        stage = getattr(event, "data", {}).get("status", "")
                        if "planning" in stage:
                            pass  # Pre-tool planning
                        elif "fallback" in stage:
                            pass  # Fell back to legacy
                    elif event.event_type == "token":
                        token = getattr(event, "data", {}).get("token", "")
                        if token:
                            answer_parts.append(token)
                    elif event.event_type == "done":
                        ans = getattr(event, "data", {}).get("answer", "")
                        if ans:
                            answer_parts.insert(0, ans)
                elif isinstance(event, dict):
                    if event.get("type") == "token":
                        answer_parts.append(event.get("token", ""))
                    elif event.get("type") == "done":
                        ans = event.get("answer", "")
                        if ans:
                            answer_parts.insert(0, ans)

        asyncio.run(_run())

        result.answer = "".join(answer_parts).strip()
        result.end_to_end_ms = (time.time() - start) * 1000.0
        result.success = bool(result.answer)

        # Try to extract stage timings from the agent's trace if available.
        if hasattr(agent, "_last_trace") and agent._last_trace:
            trace = agent._last_trace
            result.stage_timings = trace.stage_timings
            result.retrieved_chunks = trace.retrieved_chunks

    def _run_mock(self, case: Dict[str, Any], result: RAGCaseResult) -> None:
        """Generate mock data for testing the eval framework.

        This is useful for CI dry-runs where LLM API access is unavailable.
        """
        start = time.time()

        # Simulate retrieval
        result.retrieved_chunks = [
            {
                "content": f"Mock context for query: {case['query']}. "
                           f"Contains keywords: {', '.join(case.get('expected_context_keywords', []))}.",
                "score": 0.85,
                "source": "mock_collection",
            },
            {
                "content": f"Additional mock context with reference to {case.get('reference_answer', '')[:50]}.",
                "score": 0.72,
                "source": "mock_collection",
            },
        ]

        # Simulate stage timings
        result.stage_timings = {
            "retrieval": 120.5,
            "rerank": 45.3,
            "generation": 850.0,
        }

        # Simulate answer
        result.answer = case.get("reference_answer", "Mock answer.")
        result.end_to_end_ms = (time.time() - start) * 1000.0 + 1015.8  # sum of stages
        result.success = True

    def _aggregate(self, results: List[RAGCaseResult], report: RAGEvalReport) -> None:
        """Aggregate per-case results into report-level metrics."""
        successful = [r for r in results if r.success]
        report.success_count = len(successful)
        report.fail_count = len(results) - len(successful)

        if not successful:
            return

        # RAGAS metrics
        if not self.skip_ragas:
            f_scores = [r.ragas.get("faithfulness", 0) for r in successful if r.ragas]
            ar_scores = [r.ragas.get("answer_relevance", 0) for r in successful if r.ragas]
            cp_scores = [r.ragas.get("context_precision", 0) for r in successful if r.ragas]
            cr_scores = [r.ragas.get("context_recall", 0) for r in successful if r.ragas]

            n = len(successful)
            report.faithfulness = sum(f_scores) / n if f_scores else 0
            report.answer_relevance = sum(ar_scores) / n if ar_scores else 0
            report.context_precision = sum(cp_scores) / n if cp_scores else 0
            report.context_recall = sum(cr_scores) / n if cr_scores else 0
            report.ragas_average = (
                report.faithfulness + report.answer_relevance +
                report.context_precision + report.context_recall
            ) / 4.0

        # Performance metrics
        retrieval_times = [r.stage_timings.get("retrieval", 0) for r in successful if r.stage_timings]
        rerank_times = [r.stage_timings.get("rerank", 0) for r in successful if r.stage_timings]
        gen_times = [r.stage_timings.get("generation", 0) for r in successful if r.stage_timings]
        e2e_times = [r.end_to_end_ms for r in successful]

        if retrieval_times:
            report.avg_retrieval_ms = sum(retrieval_times) / len(retrieval_times)
        if rerank_times:
            report.avg_rerank_ms = sum(rerank_times) / len(rerank_times)
        if gen_times:
            report.avg_generation_ms = sum(gen_times) / len(gen_times)
        if e2e_times:
            report.avg_end_to_end_ms = sum(e2e_times) / len(e2e_times)
            e2e_sorted = sorted(e2e_times)
            p95_idx = int(len(e2e_sorted) * 0.95)
            report.p95_end_to_end_ms = e2e_sorted[min(p95_idx, len(e2e_sorted) - 1)]

    def _get_git_commit(self) -> str:
        """Get current git commit hash for baseline tracking."""
        try:
            import subprocess
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True, text=True, timeout=5,
            )
            return result.stdout.strip() if result.returncode == 0 else "unknown"
        except Exception:
            return "unknown"
