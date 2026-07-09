"""Code reviewer - layered validation with deterministic + LLM checks."""
import json
import logging
import re
from typing import List, Optional
from langchain_core.messages import HumanMessage

from code_review.report import ReviewReport, ReviewIssue, Severity
from prompts.loader import load_prompt

logger = logging.getLogger(__name__)

_REVIEW_PROMPT_FILE = "code_review_v1.txt"


SIMPLE_RULES = [
    {"name": "trailing_whitespace", "pattern": r" +$", "severity": Severity.INFO,
     "desc": "Trailing whitespace", "suggestion": "Remove trailing whitespace"},
    {"name": "tab_indent", "pattern": r"\t", "severity": Severity.INFO,
     "desc": "Tab character found", "suggestion": "Use spaces instead of tabs"},
    {"name": "long_line", "check": lambda line: len(line) > 120,
     "severity": Severity.WARNING, "desc": "Line too long (>120 chars)",
     "suggestion": "Break into multiple lines"},
    {"name": "todo", "pattern": r"# TODO|# FIXME|# HACK", "severity": Severity.INFO,
     "desc": "TODO/FIXME comment found", "suggestion": "Track in issue tracker"},
]


class CodeReviewer:
    def __init__(self, llm=None, strictness: str = "standard"):
        self.llm = llm
        self.strictness = strictness

    def review_deterministic(self, file_path: str, code: str,
                              report: ReviewReport = None):
        if report is None:
            report = ReviewReport()
        lines = code.split("\n")
        for i, line in enumerate(lines, 1):
            for rule in SIMPLE_RULES:
                matched = False
                if "pattern" in rule:
                    if re.search(rule["pattern"], line):
                        matched = True
                elif "check" in rule:
                    if rule["check"](line):
                        matched = True
                if matched:
                    report.add_issue(ReviewIssue(
                        severity=rule["severity"], file=file_path, line=i,
                        description=rule["desc"], suggestion=rule["suggestion"],
                        confidence=1.0, rule=rule["name"],
                    ))

    def review_llm(self, file_path: str, code: str, lang: str = "python",
                   report: ReviewReport = None):
        if report is None:
            report = ReviewReport()
        if not self.llm:
            return

        prompt = load_prompt(_REVIEW_PROMPT_FILE).format(lang=lang, code=code)

        try:
            import asyncio
            loop = asyncio.get_event_loop()
            if loop.is_running():
                result = self.llm.invoke([HumanMessage(content=prompt)])
            else:
                result = loop.run_until_complete(
                    self.llm.ainvoke([HumanMessage(content=prompt)])
                )

            text = result.content.strip()
            if "[" in text:
                start = text.index("[")
                end = text.rindex("]") + 1
                issues = json.loads(text[start:end])

                for issue_data in issues:
                    sev_str = issue_data.get("severity", "info")
                    severity = (Severity.ERROR if sev_str == "error"
                                 else Severity.WARNING if sev_str == "warning"
                                 else Severity.INFO)
                    confidence = float(issue_data.get("confidence", 0.5))

                    if self.strictness == "loose" and severity == Severity.WARNING:
                        severity = Severity.INFO
                    elif self.strictness == "strict" and severity == Severity.INFO:
                        continue
                    if confidence < 0.5:
                        severity = Severity.INFO

                    report.add_issue(ReviewIssue(
                        severity=severity, file=file_path,
                        line=int(issue_data.get("line", 0)),
                        description=issue_data.get("description", ""),
                        suggestion=issue_data.get("suggestion", ""),
                        confidence=confidence, rule="llm_semantic",
                    ))
        except Exception as e:
            logger.warning("LLM review failed for %s: %s", file_path, e)

    def review(self, file_path: str, code: str, lang: str = "python") -> ReviewReport:
        report = ReviewReport()
        report.files_reviewed = 1
        report.total_lines = len(code.split("\n"))
        self.review_deterministic(file_path, code, report)
        self.review_llm(file_path, code, lang, report)
        return report
