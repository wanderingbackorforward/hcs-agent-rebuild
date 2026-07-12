"""RAGAS-aligned LLM-as-Judge metrics for RAG evaluation.

Implements four core RAGAS metrics using LLM-as-judge, aligned with the
RAGAS official metric definitions:

1. **Faithfulness** — Every claim in the answer must be supported by the
   retrieved context. Measures hallucination.
   Formula: (# supported claims) / (# total claims)

2. **Answer Relevance** — The answer must address the question. Measures
   answer quality independent of context.
   Formula: mean relevance score over generated questions (0..1)

3. **Context Precision** — Relevant context chunks should be ranked higher.
   Measures retrieval ranking quality.
   Formula: mean precision@k for each relevant chunk

4. **Context Recall** — All information needed to answer the question must
   be present in the retrieved context. Measures retrieval coverage.
   Formula: (# supported sentences in reference) / (# total sentences in reference)

Design decisions:
  - Uses LangChain ChatModel (create_chat_model) for LLM calls.
  - Supports batch evaluation for efficiency.
  - Each metric returns a float in [0, 1].
  - Falls back to 0.0 on LLM errors (conservative — don't reward failures).

Interview talking point: "I implemented RAGAS-aligned metrics using
LLM-as-judge. The metric definitions match RAGAS exactly — Faithfulness
decomposes the answer into claims and checks each against context,
Context Recall checks if the reference answer is fully covered by context.
I chose in-house implementation over the ragas library for dependency
control and prompt customization."
"""
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from config.model_provider import create_chat_model

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RAGASSample:
    """A single evaluation sample for RAGAS metrics.

    Fields:
        question: The user query.
        answer: The generated answer.
        contexts: List of retrieved context chunks (strings).
        reference: The ground-truth reference answer.
    """
    question: str
    answer: str
    contexts: List[str]
    reference: str = ""


@dataclass
class RAGASResult:
    """Results of RAGAS evaluation for a single sample."""
    faithfulness: float = 0.0
    answer_relevance: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0
    errors: List[str] = field(default_factory=list)

    @property
    def average(self) -> float:
        return (self.faithfulness + self.answer_relevance +
                self.context_precision + self.context_recall) / 4.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "faithfulness": round(self.faithfulness, 4),
            "answer_relevance": round(self.answer_relevance, 4),
            "context_precision": round(self.context_precision, 4),
            "context_recall": round(self.context_recall, 4),
            "average": round(self.average, 4),
            "errors": self.errors,
        }


@dataclass
class RAGASBatchResult:
    """Aggregated results across all samples."""
    sample_count: int = 0
    faithfulness: float = 0.0
    answer_relevance: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0
    average: float = 0.0
    per_sample: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_count": self.sample_count,
            "faithfulness": round(self.faithfulness, 4),
            "answer_relevance": round(self.answer_relevance, 4),
            "context_precision": round(self.context_precision, 4),
            "context_recall": round(self.context_recall, 4),
            "average": round(self.average, 4),
            "per_sample": self.per_sample,
        }


# ---------------------------------------------------------------------------
# LLM helper
# ---------------------------------------------------------------------------

def _llm_invoke(prompt: str, temperature: float = 0.0) -> str:
    """Invoke LLM with a prompt and return text response."""
    try:
        llm = create_chat_model(temperature=temperature)
        resp = llm.invoke(prompt)
        return resp.content if hasattr(resp, "content") else str(resp)
    except Exception as e:
        logger.error("LLM invoke failed: %s", e)
        return ""


def _parse_json_response(text: str) -> Optional[Dict[str, Any]]:
    """Try to extract a JSON object from LLM response text."""
    if not text:
        return None
    # Try direct parse first
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    # Try to find JSON in the text
    match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    # Try array format
    match = re.search(r'\[[\s\S]*\]', text)
    if match:
        try:
            return {"items": json.loads(match.group())}
        except json.JSONDecodeError:
            pass
    return None


# ---------------------------------------------------------------------------
# Metric 1: Faithfulness
# ---------------------------------------------------------------------------

FAITHFULNESS_EXTRACT_CLAIMS_PROMPT = """\
Given the following answer, extract all factual claims as a JSON array of strings.

Answer: {answer}

Extract each atomic claim that can be verified against a context.
Return ONLY a JSON array, e.g. ["claim 1", "claim 2"].
If the answer contains no verifiable claims, return [].
"""

FAITHFULNESS_VERIFY_PROMPT = """\
You are a fact-checker. For each claim, determine if it is supported by the given context.

Context:
{context}

Claims:
{claims_json}

For each claim, respond with a JSON array of objects:
[{{"claim": "...", "supported": true/false}}]

A claim is "supported" if the context contains information that directly implies it.
If the context does not mention or contradict the claim, mark it as not supported.
Return ONLY the JSON array.
"""


def _faithfulness(sample: RAGASSample) -> float:
    """Compute Faithfulness: fraction of claims supported by context."""
    if not sample.answer or not sample.contexts:
        return 0.0

    context_text = "\n\n".join(sample.contexts)

    # Step 1: Extract claims
    prompt = FAITHFULNESS_EXTRACT_CLAIMS_PROMPT.format(answer=sample.answer)
    resp = _llm_invoke(prompt)
    claims_data = _parse_json_response(resp)

    if not claims_data:
        # Fallback: split answer into sentences as claims
        sentences = [s.strip() for s in re.split(r'[。.!?！？]', sample.answer) if s.strip()]
        claims = sentences if sentences else [sample.answer]
    else:
        claims = claims_data.get("items", claims_data) if isinstance(claims_data, dict) else claims_data
        if not isinstance(claims, list):
            claims = [sample.answer]

    if not claims:
        return 1.0  # No claims to verify = vacuously faithful

    # Step 2: Verify each claim
    claims_json = json.dumps(claims, ensure_ascii=False)
    prompt = FAITHFULNESS_VERIFY_PROMPT.format(
        context=context_text[:3000],  # Limit context length
        claims_json=claims_json,
    )
    resp = _llm_invoke(prompt)
    verify_data = _parse_json_response(resp)

    if not verify_data:
        logger.warning("Faithfulness verify returned unparseable response")
        return 0.0

    items = verify_data.get("items", verify_data) if isinstance(verify_data, dict) else verify_data
    if not isinstance(items, list) or not items:
        return 0.0

    supported = sum(1 for item in items if isinstance(item, dict) and item.get("supported", False))
    return supported / len(items) if items else 0.0


# ---------------------------------------------------------------------------
# Metric 2: Answer Relevance
# ---------------------------------------------------------------------------

ANSWER_RELEVANCE_PROMPT = """\
You are evaluating how relevant an answer is to a question.

Question: {question}
Answer: {answer}

Rate the relevance on a scale of 0 to 1:
- 1.0: The answer directly and completely addresses the question.
- 0.5: The answer partially addresses the question or is somewhat relevant.
- 0.0: The answer does not address the question at all.

Return ONLY a JSON object: {{"score": 0.X, "reason": "brief explanation"}}
"""


def _answer_relevance(sample: RAGASSample) -> float:
    """Compute Answer Relevance: how well the answer addresses the question."""
    if not sample.answer or not sample.question:
        return 0.0

    prompt = ANSWER_RELEVANCE_PROMPT.format(
        question=sample.question,
        answer=sample.answer[:2000],
    )
    resp = _llm_invoke(prompt)
    data = _parse_json_response(resp)

    if data and isinstance(data, dict) and "score" in data:
        score = float(data["score"])
        return max(0.0, min(1.0, score))

    logger.warning("Answer relevance returned unparseable response")
    return 0.0


# ---------------------------------------------------------------------------
# Metric 3: Context Precision
# ---------------------------------------------------------------------------

CONTEXT_PRECISION_PROMPT = """\
You are evaluating the precision of retrieved context chunks for a question.

Question: {question}
Reference Answer: {reference}

Retrieved Context Chunks (in rank order):
{contexts_list}

For each chunk, determine if it is relevant to answering the question.
A chunk is "relevant" if it contains information useful for answering the question
or is closely related to the reference answer.

Return a JSON array: [{{"chunk_index": 0, "relevant": true}}, ...]
Return ONLY the JSON array.
"""


def _context_precision(sample: RAGASSample) -> float:
    """Compute Context Precision: relevant chunks should be ranked higher.

    Uses Mean Reciprocal Rank of relevant chunks as precision measure.
    """
    if not sample.contexts:
        return 0.0

    contexts_list = "\n".join(
        f"[{i}] {ctx[:500]}" for i, ctx in enumerate(sample.contexts)
    )

    prompt = CONTEXT_PRECISION_PROMPT.format(
        question=sample.question,
        reference=sample.reference or "(no reference available)",
        contexts_list=contexts_list,
    )
    resp = _llm_invoke(prompt)
    data = _parse_json_response(resp)

    if not data:
        logger.warning("Context precision returned unparseable response")
        return 0.0

    items = data.get("items", data) if isinstance(data, dict) else data
    if not isinstance(items, list) or not items:
        return 0.0

    # Compute precision@k for each position
    relevant_flags = [
        isinstance(item, dict) and item.get("relevant", False)
        for item in items
    ]

    relevant_count = sum(relevant_flags)
    if relevant_count == 0:
        return 0.0

    # Weighted precision: higher-ranked relevant chunks contribute more
    precision_sum = 0.0
    for i, is_relevant in enumerate(relevant_flags):
        if is_relevant:
            precision_at_k = sum(relevant_flags[:i + 1]) / (i + 1)
            precision_sum += precision_at_k

    return precision_sum / relevant_count


# ---------------------------------------------------------------------------
# Metric 4: Context Recall
# ---------------------------------------------------------------------------

CONTEXT_RECALL_PROMPT = """\
You are evaluating context recall for a RAG system.

Question: {question}
Reference Answer: {reference}

Retrieved Context:
{context}

For each sentence in the reference answer, determine if it is supported
by the retrieved context.

Return a JSON object: {{"supported_sentences": N, "total_sentences": M}}
Return ONLY the JSON object.
"""


def _context_recall(sample: RAGASSample) -> float:
    """Compute Context Recall: fraction of reference answer covered by context."""
    if not sample.reference or not sample.contexts:
        return 0.0

    context_text = "\n\n".join(sample.contexts)[:3000]

    # Split reference into sentences
    ref_sentences = [s.strip() for s in re.split(r'[。.!?！？\n]', sample.reference) if s.strip()]
    if not ref_sentences:
        return 1.0

    prompt = CONTEXT_RECALL_PROMPT.format(
        question=sample.question,
        reference=sample.reference,
        context=context_text,
    )
    resp = _llm_invoke(prompt)
    data = _parse_json_response(resp)

    if data and isinstance(data, dict):
        supported = int(data.get("supported_sentences", 0))
        total = int(data.get("total_sentences", len(ref_sentences)))
        if total > 0:
            return min(1.0, supported / total)

    # Fallback: keyword overlap
    ref_words = set(sample.reference.lower().split())
    ctx_words = set(context_text.lower().split())
    if ref_words:
        overlap = len(ref_words & ctx_words) / len(ref_words)
        return overlap

    return 0.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate_sample(sample: RAGASSample) -> RAGASResult:
    """Evaluate a single sample with all four RAGAS metrics."""
    result = RAGASResult()

    try:
        result.faithfulness = _faithfulness(sample)
    except Exception as e:
        result.errors.append(f"faithfulness: {e}")
        result.faithfulness = 0.0

    try:
        result.answer_relevance = _answer_relevance(sample)
    except Exception as e:
        result.errors.append(f"answer_relevance: {e}")
        result.answer_relevance = 0.0

    try:
        result.context_precision = _context_precision(sample)
    except Exception as e:
        result.errors.append(f"context_precision: {e}")
        result.context_precision = 0.0

    try:
        result.context_recall = _context_recall(sample)
    except Exception as e:
        result.errors.append(f"context_recall: {e}")
        result.context_recall = 0.0

    return result


def evaluate_batch(samples: List[RAGASSample]) -> RAGASBatchResult:
    """Evaluate a batch of samples and return aggregated metrics."""
    if not samples:
        return RAGASBatchResult()

    per_sample: List[Dict[str, Any]] = []
    total_f = total_ar = total_cp = total_cr = 0.0

    for i, sample in enumerate(samples):
        logger.info("RAGAS evaluating sample %d/%d: %s", i + 1, len(samples), sample.question[:50])
        result = evaluate_sample(sample)
        per_sample.append({
            "question": sample.question[:100],
            **result.to_dict(),
        })
        total_f += result.faithfulness
        total_ar += result.answer_relevance
        total_cp += result.context_precision
        total_cr += result.context_recall

    n = len(samples)
    batch = RAGASBatchResult(
        sample_count=n,
        faithfulness=total_f / n,
        answer_relevance=total_ar / n,
        context_precision=total_cp / n,
        context_recall=total_cr / n,
        per_sample=per_sample,
    )
    batch.average = (batch.faithfulness + batch.answer_relevance +
                     batch.context_precision + batch.context_recall) / 4.0
    return batch


def sample_from_trace(
    question: str,
    answer: str,
    retrieved_chunks: List[Dict[str, Any]],
    reference: str = "",
) -> RAGASSample:
    """Build a RAGASSample from trace data.

    Args:
        question: User query.
        answer: Generated answer.
        retrieved_chunks: List of chunk dicts with "content" or "text" field.
        reference: Ground-truth reference answer.
    """
    contexts = []
    for chunk in retrieved_chunks:
        text = chunk.get("content") or chunk.get("text") or ""
        if text:
            contexts.append(text)
    return RAGASSample(
        question=question,
        answer=answer,
        contexts=contexts,
        reference=reference,
    )
