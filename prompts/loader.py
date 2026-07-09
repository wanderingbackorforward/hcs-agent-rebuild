"""Prompt template loader with in-memory caching.

Centralizes prompt file loading so every module reads the prompts/
directory through a single cached entry point. Prompt files are static
at deploy time, so an unbounded LRU cache is safe — the second and all
subsequent reads of the same file come from memory with zero disk I/O.

Interview talking point: "I unified six duplicate _load_prompt_template
functions into one cached loader — prompt files are read once per
process, never per call. @lru_cache gives free hit/miss observability
and a one-line invalidate path for hot-reload during development."
"""
import logging
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

# loader.py lives inside the prompts/ directory, so its own location
# is the prompts root — no fragile parent.parent.parent chains.
_PROMPTS_DIR = Path(__file__).parent


@lru_cache(maxsize=None)
def load_prompt(name: str) -> str:
    """Load a prompt template by file name, cached for the process lifetime.

    Args:
        name: File name relative to the prompts/ directory
            (e.g. "rag_answer_v1.txt").

    Returns:
        The raw prompt template text. Call .format(...) to fill placeholders.

    Raises:
        FileNotFoundError: If the named prompt file does not exist.
    """
    path = _PROMPTS_DIR / name
    if not path.is_file():
        raise FileNotFoundError(
            "Prompt template not found: {} (available: {})".format(
                path, list_prompt_names()
            )
        )
    logger.debug("Prompt loaded from disk: %s", name)
    return path.read_text(encoding="utf-8")


def invalidate_prompt_cache() -> None:
    """Clear the in-memory prompt cache.

    Prompt files are static at deploy time, so this is mainly useful for
    hot-reload during development or tests that swap prompt files at runtime.
    """
    load_prompt.cache_clear()
    logger.debug("Prompt cache invalidated")


def get_prompt_cache_info() -> tuple:
    """Return LRU cache statistics as (hits, misses, maxsize, currsize).

    A high hit ratio after warmup confirms the cache is wired correctly.
    """
    return load_prompt.cache_info()


def list_prompt_names() -> list:
    """List available prompt template file names (sorted, .txt only)."""
    if not _PROMPTS_DIR.is_dir():
        return []
    return sorted(p.name for p in _PROMPTS_DIR.glob("*.txt"))
