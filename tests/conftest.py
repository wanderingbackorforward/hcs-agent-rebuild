"""Shared pytest configuration and fixtures.

Provides a session-scoped embedding-availability probe. Integration tests that
exercise the vector-search path (dense retrieval needs a working embedding
endpoint) depend on the ``embedding_works`` fixture so they SKIP cleanly when
the embedding service is unavailable — no key, unsupported model, network
error, paid-plan limit — instead of hard-failing on an environment
precondition.

Tests that only touch SQLite metadata / collection counts (no embedding call)
keep using the lighter ``requires_embedding`` key-presence marker and still run.
"""
import os

import pytest

# Session-level cache for the probe: True / False / None (not probed yet).
_probe: dict = {"result": None}


def embedding_available() -> bool:
    """Probe whether the embedding endpoint actually works (cached per session)."""
    if _probe["result"] is not None:
        return _probe["result"]
    has_key = bool(
        os.getenv("LLM_API_KEY")
        or os.getenv("EMBEDDING_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )
    if not has_key:
        _probe["result"] = False
        return False
    try:
        from config.model_provider import create_embedding_model
        create_embedding_model().embed_query("ping")
        _probe["result"] = True
    except Exception:
        _probe["result"] = False
    return _probe["result"]


@pytest.fixture(scope="session")
def embedding_works():
    """Skip the requesting test when the embedding service is unavailable."""
    if not embedding_available():
        pytest.skip("embedding service unavailable (no key / unsupported model)")
