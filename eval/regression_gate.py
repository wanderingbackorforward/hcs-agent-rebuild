"""Regression gate — baseline storage, diff comparison, threshold gating.

This module implements the CI/CD quality gate for RAG evaluation:

1. **Baseline storage**: After a successful run, the current metrics are saved
   as a baseline JSON file in ``eval/baselines/``.
2. **Diff comparison**: On the next run, current metrics are compared against
   the latest baseline. A diff report shows rise/fall for each metric.
3. **Threshold gating**: If any core metric drops beyond its threshold,
   the gate returns ``FAIL`` (exit code 1 in CI), blocking merge/deploy.

Default thresholds (configurable via constructor or env vars):
    RAG_GATE_FAITHFULNESS_DROP = 0.05   # Faithfulness can drop max 5%
    RAG_GATE_ANSWER_RELEVANCE_DROP = 0.05
    RAG_GATE_CONTEXT_PRECISION_DROP = 0.05
    RAG_GATE_CONTEXT_RECALL_DROP = 0.05
    RAG_GATE_LATENCY_INCREASE = 0.20    # Latency can increase max 20%

Interview talking point: "My regression gate stores baseline metrics after
each successful run. On the next run, it diffs current vs baseline. If
Faithfulness drops more than 5%, the gate fails and CI blocks the merge.
This catches regressions from chunk size changes, embedding model swaps,
or retrieval strategy modifications — all automatically, no manual review."
"""
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Default thresholds: metric → max allowed drop (as absolute fraction).
DEFAULT_THRESHOLDS: Dict[str, float] = {
    "faithfulness": 0.05,
    "answer_relevance": 0.05,
    "context_precision": 0.05,
    "context_recall": 0.05,
}

# Latency threshold: max allowed increase (as relative fraction).
DEFAULT_LATENCY_THRESHOLD: float = 0.20

# Baseline directory.
BASELINE_DIR = Path(__file__).parent / "baselines"


@dataclass
class MetricDiff:
    """Difference between current and baseline for a single metric."""
    name: str
    baseline: float
    current: float
    diff: float  # current - baseline (positive = improvement)
    pct_change: float  # percentage change
    passed: bool
    threshold: float
    direction: str = "higher_better"  # "higher_better" or "lower_better"

    def to_dict(self) -> Dict[str, Any]:
        arrow = "↑" if self.diff > 0 else ("↓" if self.diff < 0 else "=")
        return {
            "metric": self.name,
            "baseline": round(self.baseline, 4),
            "current": round(self.current, 4),
            "diff": round(self.diff, 4),
            "pct_change": f"{self.pct_change:+.1f}%",
            "arrow": arrow,
            "passed": self.passed,
            "threshold": self.threshold,
        }


@dataclass
class RegressionReport:
    """Full regression comparison report."""
    gate_result: str = "PASS"  # "PASS" or "FAIL"
    current_commit: str = ""
    baseline_commit: str = ""
    baseline_timestamp: str = ""
    metric_diffs: List[MetricDiff] = field(default_factory=list)
    latency_diffs: List[MetricDiff] = field(default_factory=list)
    failed_metrics: List[str] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gate_result": self.gate_result,
            "current_commit": self.current_commit,
            "baseline_commit": self.baseline_commit,
            "baseline_timestamp": self.baseline_timestamp,
            "metric_diffs": [d.to_dict() for d in self.metric_diffs],
            "latency_diffs": [d.to_dict() for d in self.latency_diffs],
            "failed_metrics": self.failed_metrics,
            "summary": self.summary,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


class RegressionGate:
    """Baseline storage + diff comparison + threshold gating.

    Args:
        baseline_dir: Directory for baseline JSON files.
        thresholds: Metric → max allowed drop. Defaults to DEFAULT_THRESHOLDS.
        latency_threshold: Max allowed latency increase (relative).
    """

    def __init__(
        self,
        baseline_dir: Optional[Path] = None,
        thresholds: Optional[Dict[str, float]] = None,
        latency_threshold: float = DEFAULT_LATENCY_THRESHOLD,
    ):
        self.baseline_dir = baseline_dir or BASELINE_DIR
        self.thresholds = thresholds or dict(DEFAULT_THRESHOLDS)
        self.latency_threshold = latency_threshold

    # ------------------------------------------------------------------
    # Baseline storage
    # ------------------------------------------------------------------

    def save_baseline(self, report_dict: Dict[str, Any]) -> Path:
        """Save current run's metrics as a new baseline.

        Args:
            report_dict: The RAGEvalReport.to_dict() output.

        Returns:
            Path to the saved baseline file.
        """
        self.baseline_dir.mkdir(parents=True, exist_ok=True)

        commit = report_dict.get("git_commit", "unknown")[:12]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"baseline_{timestamp}_{commit}.json"
        path = self.baseline_dir / filename

        with open(path, "w", encoding="utf-8") as f:
            json.dump(report_dict, f, ensure_ascii=False, indent=2)

        # Also update "latest.json" symlink/copy for easy lookup.
        latest_path = self.baseline_dir / "latest.json"
        with open(latest_path, "w", encoding="utf-8") as f:
            json.dump(report_dict, f, ensure_ascii=False, indent=2)

        logger.info("Baseline saved: %s", path)
        return path

    def load_latest_baseline(self) -> Optional[Dict[str, Any]]:
        """Load the most recent baseline.

        Tries ``latest.json`` first, then scans for the most recent
        ``baseline_*.json`` file.
        """
        latest_path = self.baseline_dir / "latest.json"
        if latest_path.exists():
            try:
                with open(latest_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning("Failed to load latest.json: %s", e)

        # Scan for baseline files
        baselines = sorted(
            self.baseline_dir.glob("baseline_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if baselines:
            try:
                with open(baselines[0], "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning("Failed to load %s: %s", baselines[0], e)

        return None

    # ------------------------------------------------------------------
    # Diff comparison
    # ------------------------------------------------------------------

    def compare(
        self,
        current: Dict[str, Any],
        baseline: Optional[Dict[str, Any]] = None,
    ) -> RegressionReport:
        """Compare current metrics against baseline.

        Args:
            current: Current run's RAGEvalReport.to_dict().
            baseline: Baseline dict. If None, loads from storage.

        Returns:
            RegressionReport with pass/fail result and per-metric diffs.
        """
        if baseline is None:
            baseline = self.load_latest_baseline()

        report = RegressionReport(
            current_commit=current.get("git_commit", "unknown"),
        )

        if baseline is None:
            # No baseline — first run, always pass (and save as baseline).
            report.gate_result = "PASS"
            report.summary = "No baseline found. First run — saving as baseline."
            logger.info("No baseline found; treating as first run (PASS).")
            return report

        report.baseline_commit = baseline.get("git_commit", "unknown")
        report.baseline_timestamp = baseline.get("timestamp", "")

        # Compare RAGAS metrics
        current_ragas = current.get("ragas", {})
        baseline_ragas = baseline.get("ragas", {})

        for metric_name, threshold in self.thresholds.items():
            cur_val = current_ragas.get(metric_name, 0.0)
            base_val = baseline_ragas.get(metric_name, 0.0)
            diff = cur_val - base_val
            pct = (diff / base_val * 100) if base_val > 0 else (100.0 if diff > 0 else 0.0)

            # For higher_better metrics: fail if drop exceeds threshold
            passed = diff >= -threshold

            metric_diff = MetricDiff(
                name=metric_name,
                baseline=base_val,
                current=cur_val,
                diff=diff,
                pct_change=pct,
                passed=passed,
                threshold=threshold,
                direction="higher_better",
            )
            report.metric_diffs.append(metric_diff)
            if not passed:
                report.failed_metrics.append(metric_name)

        # Compare latency metrics
        current_perf = current.get("performance", {})
        baseline_perf = baseline.get("performance", {})

        for perf_metric in ["avg_retrieval_ms", "avg_rerank_ms",
                            "avg_generation_ms", "avg_end_to_end_ms"]:
            cur_val = current_perf.get(perf_metric, 0.0)
            base_val = baseline_perf.get(perf_metric, 0.0)
            if base_val == 0:
                continue
            diff = cur_val - base_val
            pct = diff / base_val
            # For lower_better metrics: fail if increase exceeds threshold
            passed = pct <= self.latency_threshold

            metric_diff = MetricDiff(
                name=perf_metric,
                baseline=base_val,
                current=cur_val,
                diff=diff,
                pct_change=pct * 100,
                passed=passed,
                threshold=self.latency_threshold,
                direction="lower_better",
            )
            report.latency_diffs.append(metric_diff)
            if not passed:
                report.failed_metrics.append(perf_metric)

        # Gate result
        if report.failed_metrics:
            report.gate_result = "FAIL"
            report.summary = (
                f"Regression detected: {len(report.failed_metrics)} metric(s) "
                f"exceeded threshold: {', '.join(report.failed_metrics)}"
            )
        else:
            report.gate_result = "PASS"
            report.summary = "All metrics within thresholds. No regression detected."

        return report

    # ------------------------------------------------------------------
    # CI-friendly entry point
    # ------------------------------------------------------------------

    def evaluate_and_gate(
        self,
        current_report: Dict[str, Any],
        save_on_pass: bool = True,
    ) -> Tuple[RegressionReport, bool]:
        """Compare current report against baseline, return (report, passed).

        If the gate passes and save_on_pass is True, the current report is
        saved as the new baseline.

        Args:
            current_report: RAGEvalReport.to_dict() output.
            save_on_pass: Save current as new baseline when gate passes.

        Returns:
            Tuple of (RegressionReport, passed: bool).
        """
        baseline = self.load_latest_baseline()
        regression = self.compare(current_report, baseline)
        passed = regression.gate_result == "PASS"

        if passed and save_on_pass:
            self.save_baseline(current_report)

        return regression, passed

    def print_report(self, regression: RegressionReport) -> None:
        """Print a human-readable regression report to stdout."""
        print("\n" + "=" * 60)
        print(f"  RAG Regression Gate: {regression.gate_result}")
        print("=" * 60)
        print(f"  Current commit:  {regression.current_commit}")
        print(f"  Baseline commit: {regression.baseline_commit}")
        print(f"  Baseline time:   {regression.baseline_timestamp}")
        print()

        if regression.metric_diffs:
            print("  RAGAS Metrics (higher is better):")
            print(f"  {'Metric':<25} {'Baseline':>10} {'Current':>10} {'Diff':>10} {'Gate':>6}")
            print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10} {'-'*6}")
            for d in regression.metric_diffs:
                status = "PASS" if d.passed else "FAIL"
                print(f"  {d.name:<25} {d.baseline:>10.4f} {d.current:>10.4f} "
                      f"{d.diff:>+10.4f} {status:>6}")
            print()

        if regression.latency_diffs:
            print("  Performance Metrics (lower is better):")
            print(f"  {'Metric':<25} {'Baseline':>10} {'Current':>10} {'Pct':>10} {'Gate':>6}")
            print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10} {'-'*6}")
            for d in regression.latency_diffs:
                status = "PASS" if d.passed else "FAIL"
                print(f"  {d.name:<25} {d.baseline:>10.1f} {d.current:>10.1f} "
                      f"{d.pct_change:>+9.1f}% {status:>6}")
            print()

        print(f"  Summary: {regression.summary}")
        if regression.failed_metrics:
            print(f"  Failed: {', '.join(regression.failed_metrics)}")
        print("=" * 60 + "\n")
