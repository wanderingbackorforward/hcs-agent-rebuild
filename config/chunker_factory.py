"""Splitter factory: switch text chunkers by config without touching call sites."""
import os
from typing import Optional

from rag.ingestion.chunking.chunker import TextChunker


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    return int(raw) if raw else default


def _env_str(name: str, default: str) -> str:
    return os.getenv(name, default).strip() or default


def create_chunker(
    lang: Optional[str] = None,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> TextChunker:
    """Build a TextChunker.

    Defaults read from env: CHUNKER_LANG (zh|en, default zh),
    CHUNKER_SIZE (chars, default 600), CHUNKER_OVERLAP (chars, default 100).
    Callers may override per-instance.
    """
    return TextChunker(
        chunk_size=chunk_size if chunk_size is not None else _env_int("CHUNKER_SIZE", 600),
        chunk_overlap=chunk_overlap if chunk_overlap is not None else _env_int("CHUNKER_OVERLAP", 100),
        lang=lang or _env_str("CHUNKER_LANG", "zh"),
    )
