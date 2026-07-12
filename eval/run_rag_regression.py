#!/usr/bin/env python
"""CI entry point — RAG regression gate.

Usage::

    # Full live run (requires LLM API):
    python eval/run_rag_regression.py --mode live

    # Mock dry-run (no LLM API needed, tests eval framework):
    python eval/run_rag_regression.py --mode mock

    # Skip RAGAS (performance-only run):
    python eval/run_rag_regression.py --mode mock --skip-ragas

    # Custom thresholds:
    python eval/run_rag_regression.py --faithfulness-drop 0.03

Exit codes:
    0 — Gate PASSED, all metrics within thresholds.
    1 — Gate FAILED, one or more metrics exceeded threshold.
    2 — Error during evaluation (e.g., LLM API failure).

This script is designed to be called from CI/CD pipelines (GitHub Actions,
GitLab CI, etc.) as a quality gate for RAG strategy changes.

Typical CI workflow::

    on: [pull_request]
    jobs:
      rag-regression:
        steps:
          - run: python eval/run_rag_regression.py --mode live
"""
import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path when run as a script.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.rag_evaluator import RAGEvaluator
from eval.regression_gate import RegressionGate


def main():
    parser = argparse.ArgumentParser(
        description="RAG regression gate — CI quality gate for RAG evaluation."
    )
    parser.add_argument(
        "--mode", choices=["live", "mock"], default="mock",
        help="Evaluation mode: 'live' (real pipeline, needs LLM API) or 'mock' (synthetic data).",
    )
    parser.add_argument(
        "--skip-ragas", action="store_true",
        help="Skip RAGAS LLM-as-judge evaluation (performance-only run).",
    )
    parser.add_argument(
        "--save-on-pass", action="store_true", default=True,
        help="Save current metrics as new baseline when gate passes (default: True).",
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Do not save baseline even on pass (one-off check).",
    )
    parser.add_argument(
        "--faithfulness-drop", type=float, default=0.05,
        help="Max allowed Faithfulness drop (default: 0.05 = 5%).",
    )
    parser.add_argument(
        "--answer-relevance-drop", type=float, default=0.05,
        help="Max allowed Answer Relevance drop (default: 0.05).",
    )
    parser.add_argument(
        "--context-precision-drop", type=float, default=0.05,
        help="Max allowed Context Precision drop (default: 0.05).",
    )
    parser.add_argument(
        "--context-recall-drop", type=float, default=0.05,
        help="Max allowed Context Recall drop (default: 0.05).",
    )
    parser.add_argument(
        "--latency-increase", type=float, default=0.20,
        help="Max allowed latency increase as fraction (default: 0.20 = 20%).",
    )
    parser.add_argument(
        "--output", type=str, default="",
        help="Path to save the full report JSON (optional).",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging.",
    )

    args = parser.parse_args()

    # Configure logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print(f"\n{'='*60}")
    print(f"  RAG Regression Gate — Starting")
    print(f"  Mode: {args.mode}")
    print(f"  Skip RAGAS: {args.skip_ragas}")
    print(f"{'='*60}\n")

    # Step 1: Run RAG evaluation
    print("[1/3] Running RAG evaluation...")
    evaluator = RAGEvaluator(
        mode=args.mode,
        skip_ragas=args.skip_ragas,
    )

    try:
        report = evaluator.run()
    except Exception as e:
        print(f"\nFATAL: Evaluation failed: {e}")
        return 2

    print(f"\n  Evaluation complete:")
    print(f"    Cases: {report.case_count} ({report.success_count} passed, {report.fail_count} failed)")
    if not args.skip_ragas:
        print(f"    Faithfulness:      {report.faithfulness:.4f}")
        print(f"    Answer Relevance:  {report.answer_relevance:.4f}")
        print(f"    Context Precision: {report.context_precision:.4f}")
        print(f"    Context Recall:    {report.context_recall:.4f}")
        print(f"    RAGAS Average:     {report.ragas_average:.4f}")
    print(f"    Avg Retrieval:  {report.avg_retrieval_ms:.1f}ms")
    print(f"    Avg Rerank:     {report.avg_rerank_ms:.1f}ms")
    print(f"    Avg Generation: {report.avg_generation_ms:.1f}ms")
    print(f"    Avg End-to-End: {report.avg_end_to_end_ms:.1f}ms")
    print(f"    P95 End-to-End: {report.p95_end_to_end_ms:.1f}ms")

    # Save full report if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report.to_json())
        print(f"\n  Full report saved: {output_path}")

    # Step 2: Compare against baseline
    print(f"\n[2/3] Comparing against baseline...")
    thresholds = {
        "faithfulness": args.faithfulness_drop,
        "answer_relevance": args.answer_relevance_drop,
        "context_precision": args.context_precision_drop,
        "context_recall": args.context_recall_drop,
    }
    gate = RegressionGate(
        thresholds=thresholds,
        latency_threshold=args.latency_increase,
    )

    save_on_pass = args.save_on_pass and not args.no_save
    regression, passed = gate.evaluate_and_gate(
        report.to_dict(),
        save_on_pass=save_on_pass,
    )

    # Step 3: Print gate result
    print(f"\n[3/3] Gate result:")
    gate.print_report(regression)

    # Exit code
    if passed:
        print("GATE: PASS — all metrics within thresholds.\n")
        return 0
    else:
        print("GATE: FAIL — regression detected, blocking merge.\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
