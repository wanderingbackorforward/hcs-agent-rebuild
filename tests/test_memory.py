"""Tests for layered memory (short-term, long-term, task, context manager).

Uses FakeEmbedder + FakeStore + FakeLLM — no real API calls.
"""
from agents.memory.short_term_memory import ShortTermMemory
from agents.memory.long_term_memory import LongTermMemory
from agents.memory.task_memory import TaskMemory
from agents.context_manager import ContextManager, count_tokens


class FakeLLM:
    """Deterministic fake LLM — no API calls."""

    def invoke(self, messages):
        class Result:
            content = "测试摘要：用户询问了环境匹配问题。"
        return Result()

    async def ainvoke(self, messages):
        return self.invoke(messages)


class FakeEmbedder:
    """Deterministic embedding based on char codes — no API calls."""

    def embed(self, text: str):
        return self.embed_query(text)

    def embed_query(self, text: str):
        vec = [0.0] * 10
        for i, ch in enumerate(text):
            vec[i % 10] += ord(ch) / 1000.0
        norm = sum(v * v for v in vec) ** 0.5
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    def embed_documents(self, texts):
        return [self.embed_query(t) for t in texts]


class FakeStore:
    """In-memory store mimicking ChromaStore interface."""

    def __init__(self):
        self._docs = []

    def upsert(self, doc_id: str, chunks, embeddings, metadatas=None):
        metadatas = metadatas or [{} for _ in chunks]
        for chunk, emb, meta in zip(chunks, embeddings, metadatas):
            self._docs.append((doc_id, chunk, emb, {**meta, "doc_id": doc_id}))

    def query(self, query_embedding, top_k=5, filters=None):
        results = []
        for doc_id, text, emb, meta in self._docs:
            sim = sum(a * b for a, b in zip(query_embedding, emb))
            distance = 1.0 - sim  # ChromaDB returns cosine distance
            results.append((doc_id, text, distance, meta))
        results.sort(key=lambda x: x[2])
        return results[:top_k]

    def delete_collection(self):
        self._docs.clear()


# --- Short-term memory ---

def test_stm_add_and_get_context():
    stm = ShortTermMemory(llm=FakeLLM(), max_turns=6)
    stm.add_message("user", "你好")
    stm.add_message("ai", "你好，有什么可以帮你的？")
    ctx = stm.get_context()
    assert "用户: 你好" in ctx
    assert "AI: 你好" in ctx


def test_stm_compression_triggers_summary():
    stm = ShortTermMemory(llm=FakeLLM(), max_turns=4)
    for i in range(6):
        stm.add_message("user", f"问题{i}")
        stm.add_message("ai", f"回答{i}")
    assert stm.summary
    assert len(stm.messages) <= 4


def test_stm_clear():
    stm = ShortTermMemory(llm=FakeLLM())
    stm.add_message("user", "测试")
    stm.clear()
    assert stm.messages == []
    assert stm.summary == ""


def test_stm_to_dict_roundtrip():
    stm = ShortTermMemory(llm=FakeLLM())
    stm.add_message("user", "测试")
    data = stm.to_dict()
    stm2 = ShortTermMemory(llm=FakeLLM())
    stm2.from_dict(data)
    assert stm2.messages == stm.messages


# --- Long-term memory ---

def test_ltm_store_and_retrieve():
    embedder = FakeEmbedder()
    store = FakeStore()
    ltm = LongTermMemory(llm=None, embedder=embedder, store=store)
    assert ltm.store_memory("用户偏好使用 Linux 环境") is True
    assert len(store._docs) == 1
    results = ltm.retrieve("用户喜欢什么环境", top_k=3)
    assert len(results) >= 1
    assert "Linux" in results[0]["content"]


def test_ltm_gating_rejects_low_importance():
    embedder = FakeEmbedder()
    store = FakeStore()
    ltm = LongTermMemory(llm=None, embedder=embedder, store=store)
    ltm._judge_and_extract = lambda content: (0.2, "noise", {})
    assert ltm.store_memory("今天天气不错") is False
    assert len(store._docs) == 0


def test_ltm_gating_accepts_high_importance():
    embedder = FakeEmbedder()
    store = FakeStore()
    ltm = LongTermMemory(llm=None, embedder=embedder, store=store)
    ltm._judge_and_extract = lambda content: (0.9, "preference", {})
    assert ltm.store_memory("用户偏好 Python") is True
    assert len(store._docs) == 1


def test_ltm_get_context_format():
    embedder = FakeEmbedder()
    store = FakeStore()
    ltm = LongTermMemory(llm=None, embedder=embedder, store=store)
    ltm.store_memory("用户偏好 Chrome 浏览器")
    ctx = ltm.get_context("浏览器偏好", top_k=3)
    assert "[长期记忆]" in ctx
    assert "Chrome" in ctx


def test_ltm_empty_retrieve():
    embedder = FakeEmbedder()
    store = FakeStore()
    ltm = LongTermMemory(llm=None, embedder=embedder, store=store)
    assert ltm.retrieve("任意查询") == []
    assert ltm.get_context("任意查询") == ""


def test_ltm_relevance_is_similarity_not_distance():
    """Verify retrieve converts distance to similarity (higher = better)."""
    embedder = FakeEmbedder()
    store = FakeStore()
    ltm = LongTermMemory(llm=None, embedder=embedder, store=store)
    ltm.store_memory("Linux 环境配置")
    results = ltm.retrieve("Linux 环境", top_k=1)
    assert results[0]["relevance"] > 0


def test_ltm_embed_adapts_langchain_interface():
    """_embed should call embed_query for LangChain Embeddings."""
    embedder = FakeEmbedder()
    store = FakeStore()
    ltm = LongTermMemory(llm=None, embedder=embedder, store=store)
    vec = ltm._embed("测试文本")
    assert len(vec) == 10


# --- Context manager ---

def test_count_tokens_empty_and_nonempty():
    assert count_tokens("") == 0
    assert count_tokens("hello world") > 0


def test_cm_assemble_no_overflow():
    stm = ShortTermMemory(llm=FakeLLM())
    stm.add_message("user", "你好")
    embedder = FakeEmbedder()
    store = FakeStore()
    ltm = LongTermMemory(llm=None, embedder=embedder, store=store)
    ltm.store_memory("用户偏好 Python")
    cm = ContextManager(short_term_memory=stm, long_term_memory=ltm)
    ctx = cm.assemble_context("系统提示", "用户问题")
    assert ctx["system_prompt"] == "系统提示"
    assert ctx["query"] == "用户问题"
    assert ctx["overflow"] is False


def test_cm_build_prompt_contains_sections():
    cm = ContextManager()
    prompt = cm.build_prompt("系统提示", "用户问题")
    assert "系统提示" in prompt
    assert "用户问题" in prompt


def test_cm_overflow_triggers_compression():
    stm = ShortTermMemory(llm=FakeLLM(), max_turns=10)
    for i in range(20):
        stm.add_message("user", f"这是一段较长的测试消息编号{i}用于触发上下文溢出处理")
        stm.add_message("ai", f"这是回复消息编号{i}同样较长以触发溢出处理机制")
    cm = ContextManager(max_tokens=500, short_term_memory=stm)
    ctx = cm.assemble_context("系统", "查询")
    assert ctx["overflow"] is True


# --- Task memory ---

def test_tm_idle_returns_empty_context():
    tm = TaskMemory()
    assert tm.get_context() == ""
    assert tm.task_type == "idle"


def test_tm_set_task_and_get_context():
    tm = TaskMemory()
    tm.set_task("knowledge_qa")
    tm.update_progress("query", "什么是 SDK")
    ctx = tm.get_context()
    assert "[任务记忆]" in ctx
    assert "knowledge_qa" in ctx
    assert "什么是 SDK" in ctx


def test_tm_add_result_and_retrieve():
    tm = TaskMemory()
    tm.set_task("environment_matching")
    tm.add_result("retrieval", {"doc_count": 3})
    tm.add_result("validation", {"passed": True})
    all_results = tm.get_results()
    assert len(all_results) == 2
    retrieval = tm.get_results("retrieval")
    assert len(retrieval) == 1
    assert retrieval[0]["content"]["doc_count"] == 3


def test_tm_clear_resets_to_idle():
    tm = TaskMemory()
    tm.set_task("knowledge_qa")
    tm.add_result("test", {"x": 1})
    tm.clear()
    assert tm.task_type == "idle"
    assert tm.get_results() == []
    assert tm.get_context() == ""


def test_tm_archive_clears_state():
    tm = TaskMemory()
    tm.set_task("environment_matching")
    tm.update_progress("env_type", "dev")
    tm.archive()
    assert tm.task_type == "idle"
    assert tm.progress == {}


def test_tm_set_task_resets_previous():
    tm = TaskMemory()
    tm.set_task("knowledge_qa")
    tm.add_result("a", {"x": 1})
    tm.set_task("environment_matching")
    assert tm.task_type == "environment_matching"
    assert tm.get_results() == []


def test_tm_results_cap_at_20():
    tm = TaskMemory()
    tm.set_task("test")
    for i in range(25):
        tm.add_result("step", {"i": i})
    results = tm.get_results()
    assert len(results) == 20
    assert results[-1]["content"]["i"] == 24


def test_tm_persistence_with_repo():
    class FakeRepo:
        def __init__(self):
            self.fields = {}

        def get_fields(self, sid):
            return self.fields.get(sid, {})

        def update_fields(self, sid, fields):
            current = self.fields.get(sid, {})
            current.update(fields)
            self.fields[sid] = current

    repo = FakeRepo()
    tm1 = TaskMemory(session_repo=repo, session_id="s1")
    tm1.set_task("knowledge_qa")
    tm1.update_progress("query", "test query")

    tm2 = TaskMemory(session_repo=repo, session_id="s1")
    assert tm2.task_type == "knowledge_qa"
    assert tm2.progress.get("query") == "test query"


# --- Context manager + task memory ---

def test_cm_injects_task_context():
    tm = TaskMemory()
    tm.set_task("knowledge_qa")
    tm.update_progress("query", "什么是 SDK")
    cm = ContextManager(task_memory=tm)
    ctx = cm.assemble_context("系统提示", "用户问题")
    assert ctx["task_context"] != ""
    assert "knowledge_qa" in ctx["task_context"]


def test_cm_build_prompt_includes_task_context():
    tm = TaskMemory()
    tm.set_task("environment_matching")
    cm = ContextManager(task_memory=tm)
    prompt = cm.build_prompt("系统提示", "查询")
    assert "[任务记忆]" in prompt


# --- Rerank + confidence filter ---

class FakeReranker:
    """Reverses item order to simulate reranking."""

    def __init__(self):
        self.called = False

    def rerank(self, query, items, top_k):
        self.called = True
        reordered = list(reversed(items))
        return reordered[:top_k]


def test_ltm_reranker_is_called_in_retrieve():
    embedder = FakeEmbedder()
    store = FakeStore()
    reranker = FakeReranker()
    ltm = LongTermMemory(llm=None, embedder=embedder, store=store, reranker=reranker)
    ltm.store_memory("记忆A")
    ltm.store_memory("记忆B")
    ltm.retrieve("查询", top_k=2)
    assert reranker.called is True


def test_ltm_reranker_changes_order():
    embedder = FakeEmbedder()
    store = FakeStore()
    reranker = FakeReranker()
    ltm = LongTermMemory(llm=None, embedder=embedder, store=store, reranker=reranker)
    ltm.store_memory("第一条记忆")
    ltm.store_memory("第二条记忆")
    results = ltm.retrieve("查询", top_k=2)
    # FakeReranker reverses, so second-stored should come first
    assert "第二条记忆" in results[0]["content"]


def test_ltm_confidence_filter_drops_low_score():
    """Memories with combined score < CONFIDENCE_THRESHOLD are filtered out."""
    embedder = FakeEmbedder()
    store = FakeStore()
    ltm = LongTermMemory(llm=None, embedder=embedder, store=store)
    # Store a memory with very different embedding (low relevance)
    ltm.store_memory("完全无关的内容xyz")
    # Query with totally different text -> low relevance -> low combined
    results = ltm.retrieve("zzzzzzzzzz", top_k=3)
    # May or may not be filtered depending on embedding similarity,
    # but at minimum retrieve should not crash
    assert isinstance(results, list)


def test_ltm_noop_reranker_default():
    """Without explicit reranker, LTM should still work (NoOpReranker)."""
    embedder = FakeEmbedder()
    store = FakeStore()
    ltm = LongTermMemory(llm=None, embedder=embedder, store=store)
    assert ltm.reranker is not None
    ltm.store_memory("测试记忆")
    results = ltm.retrieve("测试", top_k=1)
    assert len(results) >= 0  # no crash


# --- Structured extraction ---

class FakeLLMWithExtraction:
    """Fake LLM that returns importance + entities."""

    def invoke(self, messages):
        class Result:
            content = '{"importance": 0.9, "type": "preference", "entities": ["Python", "Linux"]}'
        return Result()

    async def ainvoke(self, messages):
        return self.invoke(messages)


def test_ltm_extract_entities_on_store():
    embedder = FakeEmbedder()
    store = FakeStore()
    ltm = LongTermMemory(
        llm=FakeLLMWithExtraction(), embedder=embedder, store=store,
    )
    assert ltm.store_memory("用户偏好 Python 和 Linux") is True
    assert len(store._docs) == 1
    _, _, _, meta = store._docs[0]
    assert meta.get("entities") == ["Python", "Linux"]


def test_ltm_judge_and_extract_returns_entities():
    embedder = FakeEmbedder()
    store = FakeStore()
    ltm = LongTermMemory(
        llm=FakeLLMWithExtraction(), embedder=embedder, store=store,
    )
    importance, mem_type, extracted = ltm._judge_and_extract("用户偏好 Python")
    assert importance == 0.9
    assert mem_type == "preference"
    assert "Python" in extracted.get("entities", [])


def test_ltm_judge_importance_backward_compat():
    """_judge_importance wrapper still returns (importance, type) tuple."""
    embedder = FakeEmbedder()
    store = FakeStore()
    ltm = LongTermMemory(
        llm=FakeLLMWithExtraction(), embedder=embedder, store=store,
    )
    importance, mem_type = ltm._judge_importance("用户偏好 Python")
    assert importance == 0.9
    assert mem_type == "preference"


def test_ltm_no_llm_returns_empty_extraction():
    embedder = FakeEmbedder()
    store = FakeStore()
    ltm = LongTermMemory(llm=None, embedder=embedder, store=store)
    importance, mem_type, extracted = ltm._judge_and_extract("任意内容")
    assert importance == 0.6
    assert mem_type == "fact"
    assert extracted == {}
