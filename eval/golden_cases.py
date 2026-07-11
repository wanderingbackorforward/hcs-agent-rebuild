"""Golden evaluation set — labeled cases aligned to the HCS platform's two agents.

Each case carries everything the 5 metrics need:

- ``query``              — user input
- ``expected_intent``    — knowledge_qa | environment_matching | unrelated
- ``expected_keywords``  — answer must contain these (任务成功率)
- ``expected_tools``     — tools that *should* be called (工具调用)
- ``reference_answer``   — ground-truth answer (回答质量 accuracy)
- ``require_final``      — must reach a Final Answer (执行轨迹)
- ``expected_trajectory``— optional replay trace for offline-no-LLM mode

The replay traces let the offline evaluator score all 5 metrics WITHOUT any
LLM API key — ideal for CI. In live mode the agent generates real traces.
"""
from typing import Any, Dict, List

from eval.trace import (
    AgentTrace, TraceStep, ToolCallRecord,
    REASON_FINAL_ANSWER, REASON_MAX_ITERATIONS,
)

# Schema for the 3 MCP tools (mirrors mcp_server/tools/* declarations).
TOOL_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "query_knowledge_hub": {
        "required": ["query"],
        "properties": {
            "query": {"type": "string"},
            "top_k": {"type": "integer"},
            "collection": {"type": "string"},
            "category": {"type": "string"},
            "doc_id": {"type": "string"},
            "return_mode": {"type": "string"},
            "include_scores": {"type": "boolean"},
            "include_metadata": {"type": "boolean"},
            "max_chars_per_chunk": {"type": "integer"},
        },
    },
    "list_collections": {"required": [], "properties": {}},
    "get_document_summary": {
        "required": ["doc_id"],
        "properties": {"doc_id": {"type": "string"}},
    },
}


def _golden_cases() -> List[Dict[str, Any]]:
    return [
        {
            "id": "kq-1",
            "query": "HCS 是什么？",
            "expected_intent": "knowledge_qa",
            "expected_keywords": ["混合云", "hybrid"],
            "expected_tools": ["query_knowledge_hub"],
            "reference_answer": "HCS 是混合云解决方案，用于测试环境管理。",
            "require_final": True,
        },
        {
            "id": "kq-2",
            "query": "SDK 文档里怎么初始化客户端？",
            "expected_intent": "knowledge_qa",
            "expected_keywords": ["初始化", "client", "init"],
            "expected_tools": ["query_knowledge_hub"],
            "reference_answer": "使用 SDK 的 init 方法初始化客户端，需传入配置参数。",
            "require_final": True,
        },
        {
            "id": "kq-3",
            "query": "测试规范里对环境隔离有什么要求？",
            "expected_intent": "knowledge_qa",
            "expected_keywords": ["隔离", "isolation"],
            "expected_tools": ["query_knowledge_hub"],
            "reference_answer": "测试环境要求网络与数据隔离，避免互相影响。",
            "require_final": True,
        },
        {
            "id": "env-1",
            "query": "我需要一个有 HBase 组件的测试环境",
            "expected_intent": "environment_matching",
            "expected_keywords": ["hbase", "环境"],
            "expected_tools": [],
            "reference_answer": "已为你筛选包含 HBase 组件的候选测试环境。",
            "require_final": True,
        },
        {
            "id": "env-2",
            "query": "有没有正常运行的 Linux 节点可以探测",
            "expected_intent": "environment_matching",
            "expected_keywords": ["linux", "节点"],
            "expected_tools": [],
            "reference_answer": "已匹配到可用的 Linux 节点并完成探测。",
            "require_final": True,
        },
        {
            "id": "unr-1",
            "query": "今天天气怎么样？",
            "expected_intent": "unrelated",
            "expected_keywords": [],
            "expected_tools": [],
            "reference_answer": "抱歉，我只能协助 HCS 测试相关的问题。",
            "require_final": True,
        },
    ]


def get_cases() -> List[Dict[str, Any]]:
    """Return the golden case list (fresh copy each call)."""
    import copy
    return copy.deepcopy(_golden_cases())


# --------------------------------------------------------------------------- #
# Replay traces — synthetic traces for offline scoring without an LLM.
# Used by CI and the offline evaluator's no-key path.
# --------------------------------------------------------------------------- #

def _mk(query, answer, ctx, reason, success, pt, ct, dur, steps_spec):
    """Compact trace builder. Each step_spec: (thought, tool, args, ok, err, final)."""
    tr = AgentTrace(query=query, final_answer=answer, retrieved_context=ctx,
                    success=success, termination_reason=reason,
                    prompt_tokens=pt, completion_tokens=ct)
    tr.ended_at = tr.started_at + dur
    for i, (thought, tool, args, ok, err, final) in enumerate(steps_spec, 1):
        tc = None
        if tool:
            tc = ToolCallRecord(step=i, tool_name=tool, args=args or {},
                                result="...", success=ok, error=err)
        tr.steps.append(TraceStep(step=i, thought=thought, tool_call=tc, is_final=final))
    return tr


def get_replay_traces() -> List[AgentTrace]:
    """Synthetic traces aligned 1:1 with the golden cases (good+bad mix)."""
    kq_hub = "query_knowledge_hub"
    return [
        # kq-1: ideal — 1 tool, grounded answer, clean finish.
        _mk("HCS 是什么？", "HCS 是混合云(hybrid cloud)解决方案，用于测试环境管理。",
            "HCS 是混合云解决方案，用于测试环境管理。", REASON_FINAL_ANSWER, True,
            350, 80, 1.2, [("需要查知识库", kq_hub, {"query": "HCS 是什么", "top_k": 5}, True, "", False),
                           ("信息充足", None, None, True, "", True)]),
        # kq-2: dead loop — same tool+args twice, then max iterations.
        _mk("SDK 文档里怎么初始化客户端？", "Agent reached max iterations.",
            "使用 SDK 的 init 方法初始化客户端。", REASON_MAX_ITERATIONS, False,
            900, 120, 6.5, [("再查一次", kq_hub, {"query": "SDK 初始化", "top_k": 5}, True, "", False),
                            ("再查一次", kq_hub, {"query": "SDK 初始化", "top_k": 5}, True, "", False),
                            ("还没够", None, None, True, "", False)]),
        # kq-3: hallucination — answer unsupported by context.
        _mk("测试规范里对环境隔离有什么要求？",
            "测试规范要求所有环境必须部署在火星数据中心，并使用量子加密通道进行隔离。",
            "测试环境要求网络与数据隔离，避免互相影响。", REASON_FINAL_ANSWER, True,
            300, 200, 1.8, [("查知识库", kq_hub, {"query": "环境隔离"}, True, "", False),
                            ("", None, None, True, "", True)]),
        # env-1: ideal env match — no tools, direct match.
        _mk("我需要一个有 HBase 组件的测试环境", "已为你筛选包含 HBase 组件的候选测试环境。",
            "", REASON_FINAL_ANSWER, True, 200, 60, 0.9,
            [("直接字段匹配", None, None, True, "", True)]),
        # env-2: similar clean run.
        _mk("有没有正常运行的 Linux 节点可以探测", "已匹配到可用的 Linux 节点并完成探测。",
            "", REASON_FINAL_ANSWER, True, 200, 60, 0.9,
            [("直接匹配", None, None, True, "", True)]),
        # unr-1: unrelated handled cleanly.
        _mk("今天天气怎么样？", "抱歉，我只能协助 HCS 测试相关的问题。",
            "", REASON_FINAL_ANSWER, True, 200, 60, 0.9,
            [("超出范围", None, None, True, "", True)]),
    ]
