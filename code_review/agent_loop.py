"""Code review agent loop - orchestrates the review pipeline.

Pipeline: static analysis -> code chunking -> LLM review per chunk ->
global aggregation -> confidence scoring -> tiered report.
"""
import os
import logging
from typing import List, Dict
from code_review.reviewer import CodeReviewer
from code_review.report import ReviewReport, ReviewIssue, Severity

from config.settings import app_settings

logger = logging.getLogger(__name__)
CHUNK_SIZE = app_settings.code_review_chunk_size
CHUNK_OVERLAP = app_settings.code_review_chunk_overlap


class CodeReviewLoop:
    def __init__(self, llm=None, strictness: str = "standard"):
        self.reviewer = CodeReviewer(llm=llm, strictness=strictness)

    def _chunk_code(self, code: str, file_path: str) -> List[Dict]:
        chunks = []
        if file_path.endswith(".py"):
            try:
                import ast
                tree = ast.parse(code)
                lines = code.split("\n")
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        start = node.lineno - 1
                        end = node.end_lineno if hasattr(node, "end_lineno") else start + 50
                        chunk = "\n".join(lines[start:end])
                        chunks.append({
                            "code": chunk, "start_line": start + 1,
                            "end_line": end, "name": node.name,
                        })
                if chunks:
                    return chunks
            except Exception:
                pass

        lines = code.split("\n")
        for i in range(0, len(lines), CHUNK_SIZE - CHUNK_OVERLAP):
            chunk_lines = lines[i:i + CHUNK_SIZE]
            chunks.append({
                "code": "\n".join(chunk_lines), "start_line": i + 1,
                "end_line": min(i + CHUNK_SIZE, len(lines)),
                "name": "lines_{}".format(i + 1),
            })
            if i + CHUNK_SIZE >= len(lines):
                break
        return chunks if chunks else [{"code": code, "start_line": 1, "end_line": len(code), "name": "full"}]

    def _detect_lang(self, file_path: str) -> str:
        ext_map = {
            ".py": "python", ".js": "javascript", ".ts": "typescript",
            ".java": "java", ".go": "go", ".rs": "rust", ".cpp": "cpp",
            ".c": "c", ".md": "markdown",
        }
        _, ext = os.path.splitext(file_path)
        return ext_map.get(ext, "text")

    def review_file(self, file_path: str, code: str) -> ReviewReport:
        lang = self._detect_lang(file_path)
        report = ReviewReport()
        report.files_reviewed = 1
        report.total_lines = len(code.split("\n"))
        self.reviewer.review_deterministic(file_path, code, report)

        chunks = self._chunk_code(code, file_path)
        for chunk in chunks:
            chunk_report = ReviewReport()
            self.reviewer.review_llm(file_path, chunk["code"], lang, chunk_report)
            offset = chunk["start_line"] - 1
            for issue in chunk_report.issues:
                if issue.line > 0:
                    issue.line += offset
                report.add_issue(issue)

        report.issues = self._deduplicate(report.issues)
        return report

    def review_diff(self, diff: str, file_paths: List[str] = None) -> ReviewReport:
        report = ReviewReport()
        current_file = None
        current_code = []

        for line in diff.split("\n"):
            if line.startswith("+++ b/"):
                if current_file and current_code:
                    code = "\n".join(current_code)
                    file_report = self.review_file(current_file, code)
                    report.issues.extend(file_report.issues)
                    report.files_reviewed += 1
                    report.total_lines += file_report.total_lines
                current_file = line[6:]
                current_code = []
            elif line.startswith("+") and not line.startswith("+++"):
                current_code.append(line[1:])
            elif current_file and not line.startswith("-"):
                current_code.append(line)

        if current_file and current_code:
            code = "\n".join(current_code)
            file_report = self.review_file(current_file, code)
            report.issues.extend(file_report.issues)
            report.files_reviewed += 1
            report.total_lines += file_report.total_lines

        return report

    def _deduplicate(self, issues: List[ReviewIssue]) -> List[ReviewIssue]:
        seen = set()
        unique = []
        for issue in issues:
            key = (issue.file, issue.line, issue.rule, issue.description[:50])
            if key not in seen:
                seen.add(key)
                unique.append(issue)
        return unique
