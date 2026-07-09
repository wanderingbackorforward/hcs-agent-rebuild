"""Error classifier — categorizes exceptions and builds user-facing suggestions.

Maps exception types to retryable/non-retryable categories with
user-friendly messages and actionable suggestions.  Used by the SSE
error event emission and the retry endpoint.

Classification is conservative: unknown errors default to
non-retryable to prevent retry storms.
"""
import re
from dataclasses import dataclass
from enum import Enum


class ErrorCategory(Enum):
    RETRYABLE = "retryable"
    NON_RETRYABLE = "non_retryable"


@dataclass
class ErrorInfo:
    """Structured error info for SSE events and retry decisions."""
    category: ErrorCategory
    error_type: str
    user_message: str
    suggestion: str
    max_retries: int = 3


# Exception type → (category, user_message, suggestion)
_RULES: dict[str, tuple] = {
    "TimeoutError": (
        ErrorCategory.RETRYABLE,
        "服务器响应超时",
        "建议重试，如持续超时请检查网络",
    ),
    "asyncioTimeoutError": (
        ErrorCategory.RETRYABLE,
        "服务器响应超时",
        "建议重试",
    ),
    "ConnectionError": (
        ErrorCategory.RETRYABLE,
        "网络连接异常",
        "请检查网络后重试",
    ),
    "RateLimitError": (
        ErrorCategory.RETRYABLE,
        "请求过于频繁",
        "请稍后重试",
    ),
    "ValueError": (
        ErrorCategory.NON_RETRYABLE,
        "输入格式有误",
        "请修改输入后重新发送",
    ),
    "ValidationError": (
        ErrorCategory.NON_RETRYABLE,
        "输入验证失败",
        "请检查输入内容",
    ),
    "AuthenticationError": (
        ErrorCategory.NON_RETRYABLE,
        "认证失败",
        "请检查 API Key 配置",
    ),
    "PermissionError": (
        ErrorCategory.NON_RETRYABLE,
        "权限不足",
        "请联系管理员",
    ),
}

# Substring patterns for exception messages that don't match type names.
_PATTERN_RULES: list[tuple[re.Pattern, tuple]] = [
    (re.compile(r"rate.?limit|429|too many requests", re.I), (
        ErrorCategory.RETRYABLE,
        "请求过于频繁",
        "请稍后重试",
    )),
    (re.compile(r"timeout|timed?\s*out", re.I), (
        ErrorCategory.RETRYABLE,
        "服务器响应超时",
        "建议重试",
    )),
    (re.compile(r"connection|network|unreachable", re.I), (
        ErrorCategory.RETRYABLE,
        "网络连接异常",
        "请检查网络后重试",
    )),
]

_DEFAULT = (
    ErrorCategory.NON_RETRYABLE,
    "处理异常",
    "请重新描述问题",
)


def classify(exc: Exception) -> ErrorInfo:
    """Classify an exception into ErrorInfo."""
    type_name = type(exc).__name__
    exc_msg = str(exc)

    # 1. Match by exception type name.
    if type_name in _RULES:
        cat, msg, sug = _RULES[type_name]
        return ErrorInfo(cat, type_name, msg, sug)

    # 2. Match by asyncio.TimeoutError (Python 3.11+ alias).
    if "TimeoutError" in type_name:
        cat, msg, sug = _RULES["TimeoutError"]
        return ErrorInfo(cat, type_name, msg, sug)

    # 3. Match by message pattern.
    for pattern, rule in _PATTERN_RULES:
        if pattern.search(exc_msg):
            cat, msg, sug = rule
            return ErrorInfo(cat, type_name, msg, sug)

    # 4. Default: non-retryable.
    cat, msg, sug = _DEFAULT
    return ErrorInfo(cat, type_name, msg, sug)
