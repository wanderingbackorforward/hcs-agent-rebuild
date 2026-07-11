import pytest

from config.sse_protocol import SSEEvent
from agents.knowledge_qa_agent import KnowledgeQAAgent


class FakeSemanticCache:
    def __init__(self):
        self.values = {}

    def get(self, query):
        return self.values.get(query)

    def set(self, query, answer):
        self.values[query] = answer


class FakeTaskMemory:
    def __init__(self):
        self.results = {}

    def set_task(self, _task):
        return None

    def update_progress(self, _key, _value):
        return None

    def add_result(self, key, value):
        self.results[key] = value

    def archive(self):
        return None


class FakeShortTermMemory:
    def __init__(self):
        self.messages = []

    def add_message(self, role, content):
        self.messages.append((role, content))

    def refresh_summary(self):
        return None

    def clear(self):
        self.messages.clear()


class FakeLongTermMemory:
    def store_memory(self, _text):
        return None


class FakeChatHistory:
    def __init__(self):
        self.records = []

    def add_user_message(self, text):
        self.records.append(("user", text))

    def add_ai_message(self, text):
        self.records.append(("ai", text))

    def clear(self):
        self.records.clear()


class FakeContextManager:
    def build_prompt(self, _system_prompt, _user_query):
        return "context"


class FakeRetriever:
    def retrieve(self, _query, top_k=None):
        return [
            ("legacy-doc", "legacy content", 0.9, {"title": "Legacy Doc"}),
        ][:top_k]


class FakeLLM:
    async def astream(self, _messages):
        class Chunk:
            def __init__(self, content):
                self.content = content
        yield Chunk("legacy ")
        yield Chunk("answer")


class FakeBroker:
    def __init__(self):
        self.calls = []

    async def list_collections(self, **kwargs):
        self.calls.append(("list_collections", kwargs))
        return {
            "collections": [{
                "name": "hcs_knowledge",
                "doc_count": 3,
                "categories": [{"name": "sdk", "doc_count": 1}],
            }]
        }

    async def query_knowledge_hub(self, **kwargs):
        self.calls.append(("query_knowledge_hub", kwargs))
        return {
            "answer": "这是检索回答",
            "retrieved_chunks": [{
                "doc_id": "hcs-sdk-quickstart",
                "title": "HCS SDK 快速入门",
                "text_preview": "SDK 文档摘要",
            }],
        }

    async def get_document_summary(self, **kwargs):
        self.calls.append(("get_document_summary", kwargs))
        return {
            "doc_id": "hcs-sdk-quickstart",
            "title": "HCS SDK 快速入门",
            "summary": "这篇文档主要介绍 HCS SDK 初始化方式。",
        }


def _make_agent():
    agent = KnowledgeQAAgent.__new__(KnowledgeQAAgent)
    agent.session_id = "s1"
    agent.initialized = True
    agent.task_memory = FakeTaskMemory()
    agent.short_term_memory = FakeShortTermMemory()
    agent.long_term_memory = FakeLongTermMemory()
    agent.chat_history = FakeChatHistory()
    agent.context_manager = FakeContextManager()
    agent.retriever = FakeRetriever()
    agent.llm = FakeLLM()
    agent.tool_broker = FakeBroker()
    return agent


@pytest.mark.asyncio
async def test_knowledge_qa_agent_uses_mcp_query_then_summary(monkeypatch):
    cache = FakeSemanticCache()
    monkeypatch.setattr("agents.knowledge_qa_agent.get_semantic_cache", lambda: cache)

    agent = _make_agent()

    async def _ensure_initialized():
        return None
    agent.ensure_initialized = _ensure_initialized

    outputs = []
    async for item in agent.process_stream("帮我看看 hcs-sdk-quickstart 这篇文档讲什么"):
        outputs.append(item)

    answer = "".join(
        item if isinstance(item, str) else item.text
        for item in outputs
    )
    assert "补充文档摘要" in answer
    assert "HCS SDK 快速入门" in answer
    assert [name for name, _ in agent.tool_broker.calls] == [
        "query_knowledge_hub",
        "get_document_summary",
    ]
    assert agent.task_memory.results["tool_path"]["path"] == "mcp_tools"


@pytest.mark.asyncio
async def test_knowledge_qa_agent_fallbacks_when_tool_path_fails(monkeypatch):
    cache = FakeSemanticCache()
    monkeypatch.setattr("agents.knowledge_qa_agent.get_semantic_cache", lambda: cache)

    agent = _make_agent()

    class FailingBroker:
        async def query_knowledge_hub(self, **kwargs):
            raise RuntimeError("tool path failed")
    agent.tool_broker = FailingBroker()

    async def _ensure_initialized():
        return None
    agent.ensure_initialized = _ensure_initialized

    outputs = []
    async for item in agent.process_stream("HCS 是什么"):
        outputs.append(item)

    statuses = [item for item in outputs if isinstance(item, SSEEvent)]
    answer = "".join(
        item if isinstance(item, str) else item.text
        for item in outputs
    )
    assert any(evt.data.get("stage") == "fallback" for evt in statuses)
    assert answer == "legacy answer"
    assert agent.task_memory.results["tool_path"]["path"] == "legacy_fallback"
