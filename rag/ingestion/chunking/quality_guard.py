"""Chunk quality guard — rule-based pre-filter + optional semantic re-split.

Cost-optimization layer sitting between the splitter and the embedder in the
ingestion pipeline. Instead of blindly embedding every chunk (the "全文档语义
切分" anti-pattern), it runs two cheap **zero-embedding** rules first:

1. **Structure-starved chunk**: a chunk longer than ``min_chunk_chars_for_check``
   yet containing fewer than ``min_separators_per_chunk`` structural separators
   (newlines / sentence-ending punctuation). Flags walls-of-text that the
   splitter failed to break apart.
2. **Multi-section chunk**: a single chunk matching two or more chapter-heading
   patterns (e.g. "第3章", "## ", "1.", "2."). Flags cross-section merges.

Only chunks flagged by these rules go through the expensive path — sentence
splitting + per-sentence embedding + adjacent-similarity re-splitting. Clean
chunks pass through untouched, so the common case costs zero embedding calls.

Interview talking point: "I added a rule-based guard so embedding is only
spent on structurally suspect chunks; for normal documents the guard is a
pure pass-through with O(n) character scans, no vector math."
"""
import logging
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


# Sentence-ending punctuation / structural separators used by rule #1.
_SEPARATOR_CHARS = "\n。.!！?？;；"

# Default chapter-heading patterns used by rule #2. Tuned for zh + en mixed
# technical docs; callers may override via the constructor.
DEFAULT_CHAPTER_PATTERNS: List[str] = [
    r"第[一二三四五六七八九十百零\d]+[章节篇部]",
    r"^#{1,6}\s",          # markdown ATX headings
    r"^\d+(\.\d+)+\s",      # 1. / 1.1 / 1.1.1 numbered headings
    r"^[一二三四五六七八九十]+、",  # 中文序号 一、二、
]


@dataclass
class ChunkAssessment:
    """Result of evaluating one chunk against the guard rules."""

    index: int
    suspicious: bool
    reason: str = ""


class ChunkQualityGuard:
    """Rule-based chunk quality filter with optional embedding re-split.

    Parameters mirror environment-driven config (see ``create_quality_guard``);
    passing ``enabled=False`` turns the guard into a no-op pass-through so the
    pipeline stays zero-cost for small documents.
    """

    def __init__(
        self,
        enabled: bool = False,
        min_chunk_chars_for_check: int = 300,
        min_separators_per_chunk: int = 2,
        chapter_patterns: Optional[List[str]] = None,
        resplit_similarity_threshold: float = 0.45,
    ):
        self.enabled = enabled
        self.min_chunk_chars_for_check = min_chunk_chars_for_check
        self.min_separators_per_chunk = min_separators_per_chunk
        self.chapter_patterns = [re.compile(p, re.MULTILINE) for p in
                                 (chapter_patterns or DEFAULT_CHAPTER_PATTERNS)]
        self.resplit_similarity_threshold = resplit_similarity_threshold

    # ------------------------------------------------------------------ #
    # Rule layer (zero embedding)
    # ------------------------------------------------------------------ #
    def _count_separators(self, text: str) -> int:
        return sum(1 for c in text if c in _SEPARATOR_CHARS)

    def _count_chapter_keywords(self, text: str) -> int:
        total = 0
        for pat in self.chapter_patterns:
            total += len(pat.findall(text))
        return total

    def assess(self, chunks: List[str]) -> List[ChunkAssessment]:
        """Evaluate every chunk with the two zero-embedding rules.

        Returns one :class:`ChunkAssessment` per input chunk, preserving order.
        """
        results: List[ChunkAssessment] = []
        for i, chunk in enumerate(chunks):
            if not self.enabled:
                results.append(ChunkAssessment(index=i, suspicious=False))
                continue
            # Rule #1: structure-starved wall-of-text.
            if (len(chunk) >= self.min_chunk_chars_for_check
                    and self._count_separators(chunk) < self.min_separators_per_chunk):
                results.append(ChunkAssessment(
                    index=i, suspicious=True,
                    reason="structure_starved",
                ))
                continue
            # Rule #2: multiple chapter headings in one chunk.
            if self._count_chapter_keywords(chunk) >= 2:
                results.append(ChunkAssessment(
                    index=i, suspicious=True,
                    reason="multi_section",
                ))
                continue
            results.append(ChunkAssessment(index=i, suspicious=False))
        return results

    def find_suspicious(self, chunks: List[str]) -> List[int]:
        """Convenience: return just the indices of suspicious chunks."""
        return [a.index for a in self.assess(chunks) if a.suspicious]

    # ------------------------------------------------------------------ #
    # Semantic re-split (only for suspicious chunks, needs an embedder)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """Split on sentence-ending punctuation / newlines, drop empties."""
        parts = re.split(r"(?<=[。.!！?？\n;；])", text)
        return [p.strip() for p in parts if p.strip()]

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(y * y for y in b) ** 0.5
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    def _semantic_resplit(self, chunk: str, embedder) -> List[str]:
        """Re-split a suspicious chunk at semantic breakpoints.

        Splits the chunk into sentences, embeds each, and cuts wherever
        adjacent-sentence similarity drops below the threshold. Falls back to
        returning the original chunk if the sentence split yields only one
        sentence or the embedder call fails.
        """
        sentences = self._split_sentences(chunk)
        if len(sentences) <= 1:
            return [chunk]
        try:
            vectors = embedder.embed_batch(sentences)
        except Exception as exc:
            logger.warning("quality_guard: embed failed, keep original chunk: %s", exc)
            return [chunk]

        sub_chunks: List[str] = []
        current = [sentences[0]]
        for i in range(1, len(sentences)):
            sim = self._cosine_similarity(vectors[i - 1], vectors[i])
            if sim < self.resplit_similarity_threshold:
                # Semantic break detected — flush current group, start a new one.
                sub_chunks.append("".join(current))
                current = [sentences[i]]
            else:
                current.append(sentences[i])
        if current:
            sub_chunks.append("".join(current))
        # Guard against degenerate empty results.
        return sub_chunks or [chunk]

    # ------------------------------------------------------------------ #
    # Public entry point used by the pipeline
    # ------------------------------------------------------------------ #
    def process(self, chunks: List[str], embedder=None) -> Tuple[List[str], List[ChunkAssessment]]:
        """Run the full guard: rule filter → optional semantic re-split.

        Returns ``(new_chunks, assessments)`` where ``assessments`` describes
        the original verdicts (handy for logging / metrics). When the guard is
        disabled or no chunk is suspicious, ``new_chunks`` is identical to the
        input and **no embedding call is made**.
        """
        assessments = self.assess(chunks)
        if not self.enabled:
            return chunks, assessments

        suspicious_indices = {a.index for a in assessments if a.suspicious}
        if not suspicious_indices:
            return chunks, assessments

        if embedder is None:
            # Rules flagged chunks but no embedder supplied — cannot re-split,
            # pass through unchanged (rules still recorded for observability).
            logger.info("quality_guard: %d suspicious chunks but no embedder, "
                        "passing through", len(suspicious_indices))
            return chunks, assessments

        new_chunks: List[str] = []
        resplit_count = 0
        for i, chunk in enumerate(chunks):
            if i in suspicious_indices:
                before = len(new_chunks)
                new_chunks.extend(self._semantic_resplit(chunk, embedder))
                resplit_count += len(new_chunks) - before
            else:
                new_chunks.append(chunk)
        logger.info("quality_guard: re-split %d suspicious chunks into %d segments",
                    len(suspicious_indices), resplit_count + len(suspicious_indices))
        return new_chunks, assessments
