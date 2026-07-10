"""Tests for ChunkQualityGuard — rule-based pre-filter + semantic re-split.

These tests are split into two groups:

* **Rule layer** (no embedding service needed): verifies the two zero-embedding
  heuristics flag the right chunks and pass clean ones through.
* **Semantic re-split layer**: uses a fake embedder with canned vectors to
  confirm suspicious chunks get re-split at similarity breakpoints, and that
  the guard degrades gracefully when the embedder errors out.
"""
import pytest

from rag.ingestion.chunking.quality_guard import ChunkQualityGuard


# ---------------------------------------------------------------------------
# Fake embedder for the semantic re-split tests — no network, deterministic.
# ---------------------------------------------------------------------------
class FakeEmbedder:
    """Returns one canned vector per sentence, in order."""

    def __init__(self, vectors):
        self._vectors = vectors
        self.calls = 0

    def embed_batch(self, texts):
        self.calls += 1
        # Return vectors round-robin so callers control the similarity matrix.
        return [self._vectors[i % len(self._vectors)] for i in range(len(texts))]


class FailingEmbedder:
    def embed_batch(self, texts):
        raise RuntimeError("embedding endpoint down")


# ---------------------------------------------------------------------------
# Rule layer
# ---------------------------------------------------------------------------
class TestRuleLayer:
    def test_disabled_guard_passes_everything_through(self):
        guard = ChunkQualityGuard(enabled=False)
        chunks = ["anything", "whatever"]
        results = guard.assess(chunks)
        assert all(not a.suspicious for a in results)
        # process() with no embedder must not raise and must return chunks unchanged.
        new_chunks, assessments = guard.process(chunks)
        assert new_chunks == chunks
        assert all(not a.suspicious for a in assessments)

    def test_clean_chunk_not_flagged(self):
        guard = ChunkQualityGuard(enabled=True, min_chunk_chars_for_check=50,
                                  min_separators_per_chunk=2)
        # Has newlines and sentence punctuation → well-structured.
        chunk = "这是第一句话。\n这是第二句话。\n这是第三句话。"
        assert guard.find_suspicious([chunk]) == []

    def test_structure_starved_chunk_flagged(self):
        guard = ChunkQualityGuard(enabled=True, min_chunk_chars_for_check=50,
                                  min_separators_per_chunk=3)
        # Long text with almost no separators.
        chunk = "这是一段很长的没有任何换行和句号分隔的文字" * 10
        results = guard.find_suspicious([chunk])
        assert results == [0]
        assessment = guard.assess([chunk])[0]
        assert assessment.reason == "structure_starved"

    def test_short_chunk_skips_structure_check(self):
        """Short chunks should not be flagged even if separator-poor."""
        guard = ChunkQualityGuard(enabled=True, min_chunk_chars_for_check=300,
                                  min_separators_per_chunk=2)
        chunk = "短文本没有分隔符"
        assert guard.find_suspicious([chunk]) == []

    def test_multi_section_chunk_flagged(self):
        guard = ChunkQualityGuard(enabled=True)
        chunk = "第一章 概述\n一些内容。\n第二章 详细设计\n更多内容。"
        results = guard.find_suspicious([chunk])
        assert results == [0]
        assessment = guard.assess([chunk])[0]
        assert assessment.reason == "multi_section"

    def test_single_section_not_flagged(self):
        guard = ChunkQualityGuard(enabled=True)
        chunk = "第一章 概述\n这是该章的内容说明。\n还有一些补充信息。"
        assert guard.find_suspicious([chunk]) == []

    def test_markdown_headings_flagged(self):
        guard = ChunkQualityGuard(enabled=True)
        chunk = "## 安装步骤\n内容。\n## 配置说明\n更多内容。"
        assert 0 in guard.find_suspicious([chunk])

    def test_mixed_chunks_only_suspicious_flagged(self):
        guard = ChunkQualityGuard(enabled=True, min_chunk_chars_for_check=50,
                                  min_separators_per_chunk=2)
        clean = "正常段落。\n有分隔符。\n合格。"
        starved = "无结构长文本" * 30
        multi = "第一章 开始\n内容。\n第二章 继续\n内容。"
        indices = guard.find_suspicious([clean, starved, multi])
        assert 1 in indices  # starved
        assert 2 in indices  # multi_section
        assert 0 not in indices  # clean


# ---------------------------------------------------------------------------
# Semantic re-split layer
# ---------------------------------------------------------------------------
class TestSemanticResplit:
    def test_no_suspicious_no_embedder_calls(self):
        """When rules pass everything, embedder must never be invoked."""
        guard = ChunkQualityGuard(enabled=True, min_chunk_chars_for_check=50,
                                  min_separators_per_chunk=1)
        embedder = FakeEmbedder([[1.0, 0.0]])
        clean = "正常。\n合格。\n没问题。"
        new_chunks, _ = guard.process([clean], embedder)
        assert new_chunks == [clean]
        assert embedder.calls == 0

    def test_suspicious_chunk_resplit_at_similarity_break(self):
        """Two groups of similar sentences split where similarity drops."""
        guard = ChunkQualityGuard(enabled=True, min_chunk_chars_for_check=10,
                                  min_separators_per_chunk=10,  # force structure-starved
                                  resplit_similarity_threshold=0.5)
        # 4 sentences: first two about topic A, last two about topic B.
        chunk = "话题A第一句。话题A第二句。话题B第一句。话题B第二句。"
        # Vectors: A's are parallel, B's are parallel, A⊥B.
        embedder = FakeEmbedder([
            [1.0, 0.0],  # A1
            [1.0, 0.0],  # A2  (sim A1-A2 = 1.0, no split)
            [0.0, 1.0],  # B1  (sim A2-B1 = 0.0 < 0.5 → split)
            [0.0, 1.0],  # B2  (sim B1-B2 = 1.0, no split)
        ])
        new_chunks, assessments = guard.process([chunk], embedder)
        assert assessments[0].suspicious
        assert embedder.calls == 1  # only the suspicious chunk was embedded
        assert len(new_chunks) == 2  # split into two semantic groups

    def test_resplit_falls_back_on_embedder_failure(self):
        guard = ChunkQualityGuard(enabled=True, min_chunk_chars_for_check=10,
                                  min_separators_per_chunk=10)
        chunk = "句子一。句子二。句子三。"
        new_chunks, assessments = guard.process([chunk], FailingEmbedder())
        assert assessments[0].suspicious
        # Embedding failed → keep original chunk unchanged.
        assert new_chunks == [chunk]

    def test_single_sentence_suspicious_not_split(self):
        """A suspicious chunk that yields only one sentence stays intact."""
        guard = ChunkQualityGuard(enabled=True, min_chunk_chars_for_check=10,
                                  min_separators_per_chunk=10)
        chunk = "没有句末标点的一整段文字就这样"
        embedder = FakeEmbedder([[1.0, 0.0]])
        new_chunks, assessments = guard.process([chunk], embedder)
        assert assessments[0].suspicious
        assert new_chunks == [chunk]

    def test_process_without_embedder_passes_through(self):
        """Suspicious chunks with no embedder supplied are kept as-is."""
        guard = ChunkQualityGuard(enabled=True, min_chunk_chars_for_check=10,
                                  min_separators_per_chunk=10)
        chunk = "无结构长文本" * 20
        new_chunks, assessments = guard.process([chunk], embedder=None)
        assert assessments[0].suspicious
        assert new_chunks == [chunk]


# ---------------------------------------------------------------------------
# Factory integration
# ---------------------------------------------------------------------------
class TestFactory:
    def test_create_quality_guard_defaults_to_disabled(self):
        from config.chunker_factory import create_quality_guard
        guard = create_quality_guard()
        # Default from settings: CHUNK_GUARD_ENABLED=False
        assert guard.enabled is False

    def test_create_quality_guard_override(self):
        from config.chunker_factory import create_quality_guard
        guard = create_quality_guard(enabled=True, min_chunk_chars_for_check=100)
        assert guard.enabled is True
        assert guard.min_chunk_chars_for_check == 100
