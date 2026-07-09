"""Tests for prompts.loader — cached prompt template loading.

Verifies that the unified loader:
- returns the same content as direct file reads,
- hits the in-memory cache on the second call (zero disk I/O),
- supports invalidation,
- lists available templates,
- raises clearly on a missing template.
"""
from pathlib import Path

import pytest

from prompts.loader import (
    get_prompt_cache_info,
    invalidate_prompt_cache,
    list_prompt_names,
    load_prompt,
)

KNOWN_PROMPT = "rag_answer_v1.txt"
_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


@pytest.fixture(autouse=True)
def _clean_cache():
    """Start every test with an empty cache so counts are isolated."""
    invalidate_prompt_cache()
    yield
    invalidate_prompt_cache()


def test_load_prompt_returns_content():
    content = load_prompt(KNOWN_PROMPT)
    assert isinstance(content, str)
    assert content.strip(), "prompt content should be non-empty"


def test_load_prompt_matches_disk():
    expected = (_PROMPTS_DIR / KNOWN_PROMPT).read_text(encoding="utf-8")
    assert load_prompt(KNOWN_PROMPT) == expected


def test_load_prompt_caches_second_call():
    first = load_prompt(KNOWN_PROMPT)
    second = load_prompt(KNOWN_PROMPT)
    hits, misses, _, size = get_prompt_cache_info()

    assert first == second
    assert misses == 1, "first call should be a cache miss"
    assert hits == 1, "second call should be a cache hit"
    assert size == 1, "exactly one entry should be cached"


def test_load_prompt_reads_disk_only_once(monkeypatch):
    """Three loads of the same name must trigger only one disk read."""
    calls = []
    real_read_text = Path.read_text

    def spy_read_text(self, *args, **kwargs):
        calls.append(self.name)
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", spy_read_text)

    name = KNOWN_PROMPT
    for _ in range(3):
        load_prompt(name)

    disk_reads = [n for n in calls if n == name]
    assert len(disk_reads) == 1, (
        "expected exactly one disk read for {}, got {}".format(name, len(disk_reads))
    )


def test_invalidate_clears_cache():
    load_prompt(KNOWN_PROMPT)
    assert get_prompt_cache_info()[3] == 1  # currsize

    invalidate_prompt_cache()
    assert get_prompt_cache_info()[3] == 0  # cleared

    # Next load must be a fresh miss, not a hit.
    load_prompt(KNOWN_PROMPT)
    hits, misses, _, size = get_prompt_cache_info()
    assert hits == 0
    assert misses == 1
    assert size == 1


def test_list_prompt_names_includes_known():
    names = list_prompt_names()
    assert isinstance(names, list)
    assert KNOWN_PROMPT in names
    # Every entry must be a .txt file.
    assert all(n.endswith(".txt") for n in names)
    # Result is sorted.
    assert names == sorted(names)


def test_load_prompt_missing_file_raises():
    with pytest.raises(FileNotFoundError) as exc_info:
        load_prompt("does_not_exist_v999.txt")
    msg = str(exc_info.value)
    assert "does_not_exist_v999.txt" in msg
    # Error message lists available templates to aid debugging.
    assert "available" in msg.lower()


@pytest.mark.parametrize("name", [n for n in (
    "classification_v1.txt",
    "code_review_v1.txt",
    "context_lock_judge_v1.txt",
    "eval_hallucination_v1.txt",
    "eval_llm_judge_v1.txt",
    "ltm_judge_and_extract_v1.txt",
    "rag_answer_v1.txt",
    "react_v1.txt",
    "stm_rolling_summary_v1.txt",
) if (_PROMPTS_DIR / n).is_file()])
def test_all_known_prompts_load(name):
    """Every checked-in prompt template must load without error."""
    content = load_prompt(name)
    assert content.strip(), "{} loaded empty".format(name)
