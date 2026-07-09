"""AI Code Review module - automated code review with layered validation.

Interview talking point: "I implemented an AI Code Review module with layered
validation — deterministic checks (lint/compile, zero false positives) + LLM
semantic review + confidence scoring. Reports are tiered: Error blocks merge,
Warning suggests, Info informs. The agent loop mirrors my document scanning
pipeline architecture."
"""
from code_review.reviewer import CodeReviewer
from code_review.agent_loop import CodeReviewLoop
from code_review.report import ReviewReport, ReviewIssue, Severity

__all__ = ["CodeReviewer", "CodeReviewLoop", "ReviewReport", "ReviewIssue", "Severity"]
