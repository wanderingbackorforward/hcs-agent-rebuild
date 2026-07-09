"""Tests for error classifier — exception categorization and retry decisions."""
import asyncio
import pytest

from config.error_classifier import (
    ErrorCategory,
    ErrorInfo,
    classify,
)


class TestErrorCategory:
    def test_retryable_value(self):
        assert ErrorCategory.RETRYABLE.value == "retryable"

    def test_non_retryable_value(self):
        assert ErrorCategory.NON_RETRYABLE.value == "non_retryable"


class TestClassifyByType:
    def test_timeout_error_retryable(self):
        exc = TimeoutError("operation timed out")
        info = classify(exc)
        assert info.category is ErrorCategory.RETRYABLE
        assert info.error_type == "TimeoutError"
        assert "超时" in info.user_message

    def test_connection_error_retryable(self):
        exc = ConnectionError("connection refused")
        info = classify(exc)
        assert info.category is ErrorCategory.RETRYABLE
        assert info.error_type == "ConnectionError"
        assert "网络" in info.user_message

    def test_value_error_non_retryable(self):
        exc = ValueError("invalid input")
        info = classify(exc)
        assert info.category is ErrorCategory.NON_RETRYABLE
        assert info.error_type == "ValueError"
        assert "输入" in info.user_message

    def test_permission_error_non_retryable(self):
        exc = PermissionError("not allowed")
        info = classify(exc)
        assert info.category is ErrorCategory.NON_RETRYABLE
        assert info.error_type == "PermissionError"
        assert "权限" in info.user_message

    def test_rate_limit_error_retryable(self):
        class RateLimitError(Exception):
            pass

        info = classify(RateLimitError("too many"))
        assert info.category is ErrorCategory.RETRYABLE
        assert info.error_type == "RateLimitError"

    def test_authentication_error_non_retryable(self):
        class AuthenticationError(Exception):
            pass

        info = classify(AuthenticationError("bad key"))
        assert info.category is ErrorCategory.NON_RETRYABLE
        assert info.error_type == "AuthenticationError"

    def test_validation_error_non_retryable(self):
        class ValidationError(Exception):
            pass

        info = classify(ValidationError("bad data"))
        assert info.category is ErrorCategory.NON_RETRYABLE
        assert info.error_type == "ValidationError"


class TestClassifyAsyncioTimeout:
    def test_asyncio_timeout_alias(self):
        """asyncio.TimeoutError is an alias for TimeoutError in 3.11+."""
        exc = asyncio.TimeoutError()
        info = classify(exc)
        assert info.category is ErrorCategory.RETRYABLE

    def test_asyncio_timeout_with_custom_name(self):
        """If type name contains 'TimeoutError', it's retryable."""
        class AsyncTimeoutError(Exception):
            pass

        info = classify(AsyncTimeoutError("task took too long"))
        assert info.category is ErrorCategory.RETRYABLE
        assert info.error_type == "AsyncTimeoutError"


class TestClassifyByPattern:
    def test_rate_limit_pattern_in_message(self):
        exc = Exception("rate limit exceeded: 429 too many requests")
        info = classify(exc)
        assert info.category is ErrorCategory.RETRYABLE
        assert "频繁" in info.user_message

    def test_rate_limit_429_pattern(self):
        exc = Exception("HTTP 429: too many requests")
        info = classify(exc)
        assert info.category is ErrorCategory.RETRYABLE

    def test_timeout_pattern_in_message(self):
        exc = RuntimeError("the request timed out after 30s")
        info = classify(exc)
        assert info.category is ErrorCategory.RETRYABLE
        assert "超时" in info.user_message

    def test_connection_pattern_in_message(self):
        exc = RuntimeError("network unreachable")
        info = classify(exc)
        assert info.category is ErrorCategory.RETRYABLE
        assert "网络" in info.user_message

    def test_timed_out_pattern(self):
        exc = RuntimeError("operation timed out")
        info = classify(exc)
        assert info.category is ErrorCategory.RETRYABLE


class TestClassifyDefault:
    def test_unknown_exception_non_retryable(self):
        exc = RuntimeError("something unexpected happened")
        info = classify(exc)
        assert info.category is ErrorCategory.NON_RETRYABLE
        assert info.error_type == "RuntimeError"
        assert info.user_message == "处理异常"

    def test_key_error_non_retryable(self):
        exc = KeyError("missing key")
        info = classify(exc)
        assert info.category is ErrorCategory.NON_RETRYABLE
        assert info.error_type == "KeyError"


class TestErrorInfoStructure:
    def test_max_retries_default(self):
        info = classify(TimeoutError("slow"))
        assert info.max_retries == 3

    def test_suggestion_not_empty(self):
        info = classify(ConnectionError("down"))
        assert info.suggestion  # truthy

    def test_suggestion_not_empty_non_retryable(self):
        info = classify(ValueError("bad"))
        assert info.suggestion  # truthy

    def test_category_string_comparison(self):
        """SSE emission checks category.value == 'retryable'."""
        info = classify(TimeoutError("t"))
        assert info.category.value == "retryable"
        info2 = classify(ValueError("v"))
        assert info2.category.value == "non_retryable"
