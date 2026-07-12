"""Regression gate — baseline storage, diff comparison, threshold gating.

This module implements the CI/CD quality gate for RAG evaluation:

1. **Baseline storage**: After a successful run, the current metrics are saved
   as a baseline JSON file in ``eval/baselines/``.
2. **Diff comparison**: On the next run, current metrics are compared against
   the latest baseline. A diff report shows rise/fall for each metric.
3. **Relative threshold gating**: If any core metric drops beyond its threshold
   relative to baseline, the gate fails.
4. **Absolute threshold gating**: Even without a baseline, metrics must meet
   absolute minimums (e.g., Faithfulness ≥ 0.7, P95 ≤ 5s). This catches
   gradual degradation that relative thresholds miss.
5. **Trade-off detection**: When quality improves but latency regresses (or
   vice versa), a trade-off warning is surfaced for manual review.

Default thresholds:
    Relative (drop from baseline):
        RAG_GATE_FAITHFULNESS_DROP = 0.05
        RAG_GATE_ANSWER_RELEVANCE_DROP = 0.05
        RAG_GATE_CONTEXT_PRECISION_DROP = 0.05
        RAG_GATE_CONTEXT_RECALL_DROP = 0.05
        RAG_GATE_LATENCY_INCREASE = 0.20

    Absolute (must meet regardless of baseline):
        FAITHFULNESS_MIN = 0.7
        ANSWER_RELEVANCE_MIN = 0.7
        CONTEXT_PRECISION_MIN = 0.7
        P95_LATENCY_MAX_MS = 5000

Interview talking point: "My regression gate has two layers: relative
thresholds catch regressions from a specific change (e.g., chunk size tweak
dropped Faithfulness by 6%), and absolute thresholds catch gradual
degradation (each step passes relative gate, but cumulative decline pushes
Faithfulness below 0.7). If quality improves but latency regresses, I surface
a trade-off warning rather than auto-blocking — the team decides."
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

# Absolute thresholds: metrics must meet these regardless of baseline.
# These catch gradual degradation that relative thresholds miss.
DEFAULT_ABSOLUTE_MIN: Dict[str, float] = {
    "faithfulness": 0.7,
    "answer_relevance": 0.7,
    "context_precision": 0.7,
    "context_recall": 0.6,  # Slightly more lenient
}

DEFAULT_ABSOLUTE_MAX_MS: Dict[str, float] = {
    "p95_end_to_end_ms": 5000.0,  # P95 latency must be ≤ 5s
}

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
class AbsoluteCheck:
    """Result of an absolute threshold check on a single metric."""
    name: str
    current: float
    threshold: float
    direction: str  # "min" (value must be ≥ threshold) or "max" (value must be ≤ threshold)
    passed: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric": self.name,
            "current": round(self.current, 4),
            "threshold": self.threshold,
            "direction": self.direction,
            "passed": self.passed,
        }


@dataclass
class TradeOffWarning:
    """Warning when quality and latency move in opposite directions."""
    description: str
    quality_metrics: List[str] = field(default_factory=list)  # metrics that improved
    latency_metrics: List[str] = field(default_factory=list)  # metrics that regressed

    def to_dict(self) -> Dict[str, Any]:
        return {
            "description": self.description,
            "quality_improved": self.quality_metrics,
            "latency_regressed": self.latency_metrics,
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
    absolute_checks: List[AbsoluteCheck] = field(default_factory=list)
    trade_offs: List[TradeOffWarning] = field(default_factory=list)
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
            "absolute_checks": [a.to_dict() for a in self.absolute_checks],
            "trade_offs": [t.to_dict() for t in self.trade_offs],
            "failed_metrics": self.failed_metrics,
            "summary": self.summary,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


class RegressionGate:
    """Baseline storage + diff comparison + threshold gating.

    Args:
        baseline_dir: Directory for baseline JSON files.
        thresholds: Metric → max allowed drop (relative). Defaults to DEFAULT_THRESHOLDS.
        latency_threshold: Max allowed latency increase (relative fraction).
        absolute_min: Metric → absolute minimum value (e.g., {"faithfulness": 0.7}).
        absolute_max_ms: Metric → absolute maximum value in ms (e.g., {"p95_end_to_end_ms": 5000}).
    """

    def __init__(
        self,
        baseline_dir: Optional[Path] = None,
        thresholds: Optional[Dict[str, float]] = None,
        latency_threshold: float = DEFAULT_LATENCY_THRESHOLD,
        absolute_min: Optional[Dict[str, float]] = None,
        absolute_max_ms: Optional[Dict[str, float]] = None,
    ):
        self.baseline_dir = baseline_dir or BASELINE_DIR
        self.thresholds = thresholds or dict(DEFAULT_THRESHOLDS)
        self.latency_threshold = latency_threshold
        self.absolute_min = absolute_min if absolute_min is not None else dict(DEFAULT_ABSOLUTE_MIN)
        self.absolute_max_ms = absolute_max_ms if absolute_max_ms is not None else dict(DEFAULT_ABSOLUTE_MAX_MS)

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
            # No baseline — first run. Still check absolute thresholds.
            self._check_absolute_thresholds(current, report)
            if report.failed_metrics:
                report.gate_result = "FAIL"
                report.summary = (
                    f"First run but absolute thresholds not met: "
                    f"{', '.join(report.failed_metrics)}"
                )
            else:
                report.gate_result = "PASS"
                report.summary = "No baseline found. First run — saving as baseline."
            logger.info("No baseline found; treating as first run (%s).", report.gate_result)
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

        # ---- Absolute threshold checks (independent of baseline) ----
        # These catch gradual degradation and first-run quality issues.
        self._check_absolute_thresholds(current, report)

        # ---- Trade-off detection ----
        # When quality improves but latency regresses (or vice versa),
        # surface a trade-off warning for manual review.
        quality_improved = [d.name for d in report.metric_diffs if d.diff > 0]
        latency_regressed = [d.name for d in report.latency_diffs if not d.passed]

        if quality_improved and latency_regressed:
            report.trade_offs.append(TradeOffWarning(
                description=(
                    f"Quality improved ({', '.join(quality_improved)}) but latency "
                    f"regressed ({', '.join(latency_regressed)}). "
                    f"Review whether the quality gain justifies the latency cost."
                ),
                quality_metrics=quality_improved,
                latency_metrics=latency_regressed,
            ))

        quality_regressed = [d.name for d in report.metric_diffs if not d.passed]
        latency_improved = [d.name for d in report.latency_diffs if d.diff < 0]

        if quality_regressed and latency_improved:
            report.trade_offs.append(TradeOffWarning(
                description=(
                    f"Latency improved ({', '.join(latency_improved)}) but quality "
                    f"regressed ({', '.join(quality_regressed)}). "
                    f"Review whether the speed gain justifies the quality loss."
                ),
                quality_metrics=quality_regressed,
                latency_metrics=latency_improved,
            ))

        # Gate result
        if report.failed_metrics:
            report.gate_result = "FAIL"
            report.summary = (
                f"Regression detected: {len(report.failed_metrics)} metric(s) "
                f"failed: {', '.join(report.failed_metrics)}"
            )
        else:
            report.gate_result = "PASS"
            if report.trade_offs:
                report.summary = (
                    "All metrics within thresholds, but trade-off detected. "
                    "Review before merging."
                )
            else:
                report.summary = "All metrics within thresholds. No regression detected."

        return report

    # ------------------------------------------------------------------
    # Absolute threshold check (shared between first-run and diff mode)
    # ------------------------------------------------------------------

    def _check_absolute_thresholds(
        self, current: Dict[str, Any], report: RegressionReport
    ) -> None:
        """Check current metrics against absolute min/max thresholds.

        These are independent of baseline — they catch gradual degradation
        and first-run quality issues.
        """
        current_ragas = current.get("ragas", {})
        current_perf = current.get("performance", {})

        for metric_name, min_val in self.absolute_min.items():
            cur_val = current_ragas.get(metric_name, 0.0)
            passed = cur_val >= min_val
            report.absolute_checks.append(AbsoluteCheck(
                name=metric_name,
                current=cur_val,
                threshold=min_val,
                direction="min",
                passed=passed,
            ))
            if not passed:
                report.failed_metrics.append(f"{metric_name} (absolute: <{min_val})")

        for metric_name, max_val in self.absolute_max_ms.items():
            cur_val = current_perf.get(metric_name, 0.0)
            if cur_val == 0:
                continue
            passed = cur_val <= max_val
            report.absolute_checks.append(AbsoluteCheck(
                name=metric_name,
                current=cur_val,
                threshold=max_val,
                direction="max",
                passed=passed,
            ))
            if not passed:
                report.failed_metrics.append(f"{metric_name} (absolute: >{max_val}ms)")

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

        if regression.absolute_checks:
            print("  Absolute Thresholds (must meet regardless of baseline):")
            print(f"  {'Metric':<25} {'Current':>10} {'Threshold':>10} {'Gate':>6}")
            print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*6}")
            for a in regression.absolute_checks:
                status = "PASS" if a.passed else "FAIL"
                op = "≥" if a.direction == "min" else "≤"
                print(f"  {a.name:<25} {a.current:>10.4f} {op} {a.threshold:>7.4f} {status:>6}")
            print()

        if regression.trade_offs:
            print("  Trade-off Warnings:")
            for t in regression.trade_offs:
                print(f"  ⚠ {t.description}")
            print()

        print(f"  Summary: {regression.summary}")
        if regression.failed_metrics:
            print(f"  Failed: {', '.join(regression.failed_metrics)}")
        print("=" * 60 + "\n")
