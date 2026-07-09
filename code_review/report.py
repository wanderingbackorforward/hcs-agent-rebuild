"""Review report and issue data models."""
from enum import Enum
from typing import List, Optional
from dataclasses import dataclass, field


class Severity(Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ReviewIssue:
    severity: Severity
    file: str
    line: int
    description: str
    suggestion: str = ""
    confidence: float = 0.0
    rule: str = ""

    def to_dict(self) -> dict:
        return {
            "severity": self.severity.value,
            "file": self.file,
            "line": self.line,
            "description": self.description,
            "suggestion": self.suggestion,
            "confidence": self.confidence,
            "rule": self.rule,
        }


@dataclass
class ReviewReport:
    issues: List[ReviewIssue] = field(default_factory=list)
    files_reviewed: int = 0
    total_lines: int = 0

    def add_issue(self, issue: ReviewIssue):
        self.issues.append(issue)

    @property
    def errors(self) -> List[ReviewIssue]:
        return [i for i in self.issues if i.severity == Severity.ERROR]

    @property
    def warnings(self) -> List[ReviewIssue]:
        return [i for i in self.issues if i.severity == Severity.WARNING]

    @property
    def infos(self) -> List[ReviewIssue]:
        return [i for i in self.issues if i.severity == Severity.INFO]

    @property
    def blocks_merge(self) -> bool:
        return len(self.errors) > 0

    def summary(self) -> str:
        return "{} errors, {} warnings, {} info ({} files reviewed)".format(
            len(self.errors), len(self.warnings), len(self.infos), self.files_reviewed)

    def to_text(self) -> str:
        lines = ["# Code Review Report", self.summary(), ""]
        if self.errors:
            lines.append("## Errors (must fix)")
            for issue in self.errors:
                lines.append("- {}: {} - {}".format(issue.file, issue.line, issue.description))
                if issue.suggestion:
                    lines.append("  Fix: {}".format(issue.suggestion))
            lines.append("")
        if self.warnings:
            lines.append("## Warnings (should fix)")
            for issue in self.warnings:
                lines.append("- {}: {} - {}".format(issue.file, issue.line, issue.description))
            lines.append("")
        if self.infos:
            lines.append("## Info (for reference)")
            for issue in self.infos:
                lines.append("- {}: {} - {}".format(issue.file, issue.line, issue.description))
        return "\n".join(lines)
