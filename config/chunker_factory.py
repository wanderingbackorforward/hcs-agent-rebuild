"""Splitter factory: switch text chunkers by config without touching call sites."""
import os
from typing import Optional

from rag.ingestion.chunking.chunker import TextChunker
from rag.ingestion.chunking.quality_guard import ChunkQualityGuard
from config.settings import app_settings


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


def create_quality_guard(
    enabled: Optional[bool] = None,
    min_chunk_chars_for_check: Optional[int] = None,
    min_separators_per_chunk: Optional[int] = None,
    resplit_similarity_threshold: Optional[float] = None,
) -> ChunkQualityGuard:
    """Build a ChunkQualityGuard from app_settings / env.

    Defaults read from CHUNK_GUARD_* env vars (see settings.AppSettings).
    Callers may override per-instance; passing ``enabled=False`` yields a
    pure pass-through guard so the pipeline stays zero-cost by default.
    """
    return ChunkQualityGuard(
        enabled=app_settings.chunk_guard_enabled if enabled is None else enabled,
        min_chunk_chars_for_check=(min_chunk_chars_for_check
                                   if min_chunk_chars_for_check is not None
                                   else app_settings.chunk_guard_min_chars),
        min_separators_per_chunk=(min_separators_per_chunk
                                  if min_separators_per_chunk is not None
                                  else app_settings.chunk_guard_min_separators),
        resplit_similarity_threshold=(resplit_similarity_threshold
                                      if resplit_similarity_threshold is not None
                                      else app_settings.chunk_guard_resplit_threshold),
    )
