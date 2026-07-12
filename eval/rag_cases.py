"""RAG-specific golden test set for KnowledgeQAAgent evaluation.

15 cases covering typical RAG scenarios:
  - Factual QA (single-hop)
  - Multi-hop reasoning
  - No-result / out-of-scope
  - Document drill-down (list → query → summary)
  - Summary request
  - Cross-collection search
  - Ambiguous / fuzzy query
  - Keyword-specific lookup
  - Error handling (invalid params)
  - Chinese + English mixed

Each case carries:
  - query: user input
  - reference_answer: ground-truth answer for RAGAS metrics
  - category: scenario category
  - expected_context_keywords: terms that should appear in retrieved context
  - min_context_relevance: minimum acceptable context recall (0..1)
"""
from typing import Any, Dict, List


def _rag_cases() -> List[Dict[str, Any]]:
    return [
        # --- Factual QA (single-hop) ---
        {
            "id": "rag-01",
            "query": "HCS 是什么？",
            "reference_answer": "HCS 是混合云解决方案，用于测试环境管理。",
            "category": "factual_qa",
            "expected_context_keywords": ["混合云", "hybrid", "HCS"],
            "min_context_relevance": 0.6,
        },
        {
            "id": "rag-02",
            "query": "SDK 怎么初始化客户端？",
            "reference_answer": "使用 SDK 的 init 方法初始化客户端，需传入配置参数。",
            "category": "factual_qa",
            "expected_context_keywords": ["init", "客户端", "client", "SDK"],
            "min_context_relevance": 0.5,
        },
        {
            "id": "rag-03",
            "query": "测试规范对环境隔离有什么要求？",
            "reference_answer": "测试环境要求网络与数据隔离，避免互相影响。",
            "category": "factual_qa",
            "expected_context_keywords": ["隔离", "isolation", "环境"],
            "min_context_relevance": 0.5,
        },
        # --- Multi-hop reasoning ---
        {
            "id": "rag-04",
            "query": "HCS 平台支持哪些组件，HBase 环境需要什么配置？",
            "reference_answer": "HCS 支持多种大数据组件，HBase 环境需要配置 HDFS 和 Zookeeper。",
            "category": "multi_hop",
            "expected_context_keywords": ["HBase", "组件", "HDFS", "Zookeeper"],
            "min_context_relevance": 0.4,
        },
        {
            "id": "rag-05",
            "query": "SDK 初始化后怎么创建测试环境并部署应用？",
            "reference_answer": "SDK 初始化后通过 create_environment 方法创建环境，再用 deploy 方法部署应用。",
            "category": "multi_hop",
            "expected_context_keywords": ["create", "environment", "deploy", "SDK"],
            "min_context_relevance": 0.4,
        },
        # --- No-result / out-of-scope ---
        {
            "id": "rag-06",
            "query": "HCS 支持量子计算吗？",
            "reference_answer": "知识库中没有关于量子计算的相关信息。",
            "category": "no_result",
            "expected_context_keywords": [],
            "min_context_relevance": 0.0,
        },
        {
            "id": "rag-07",
            "query": "今天股市行情怎么样？",
            "reference_answer": "抱歉，我只能协助 HCS 测试相关的问题。",
            "category": "out_of_scope",
            "expected_context_keywords": [],
            "min_context_relevance": 0.0,
        },
        # --- Document drill-down ---
        {
            "id": "rag-08",
            "query": "有哪些知识库文档可以查？",
            "reference_answer": "知识库包含 SDK 文档、测试规范、环境配置指南等分类。",
            "category": "discovery",
            "expected_context_keywords": ["SDK", "测试", "环境"],
            "min_context_relevance": 0.3,
        },
        {
            "id": "rag-09",
            "query": "hcs-sdk-quickstart 这篇文档讲了什么？",
            "reference_answer": "HCS SDK 快速入门文档介绍了 SDK 的安装、初始化和基本使用方法。",
            "category": "doc_summary",
            "expected_context_keywords": ["SDK", "快速入门", "init"],
            "min_context_relevance": 0.5,
        },
        # --- Summary request ---
        {
            "id": "rag-10",
            "query": "给我总结一下测试规范文档的要点",
            "reference_answer": "测试规范要求环境隔离、数据隔离，并有明确的测试流程和质量标准。",
            "category": "doc_summary",
            "expected_context_keywords": ["测试规范", "隔离", "流程"],
            "min_context_relevance": 0.4,
        },
        # --- Ambiguous / fuzzy ---
        {
            "id": "rag-11",
            "query": "怎么部署",
            "reference_answer": "部署应用需要先创建测试环境，然后通过 SDK 或平台界面进行部署。",
            "category": "ambiguous",
            "expected_context_keywords": ["部署", "deploy", "环境"],
            "min_context_relevance": 0.3,
        },
        {
            "id": "rag-12",
            "query": "环境配置相关的文档在哪",
            "reference_answer": "环境配置相关文档可以在知识库的环境配置分类中找到。",
            "category": "ambiguous",
            "expected_context_keywords": ["环境", "配置", "文档"],
            "min_context_relevance": 0.3,
        },
        # --- Keyword-specific lookup ---
        {
            "id": "rag-13",
            "query": "HDFS 配置参数 dfs.replication 默认值是多少？",
            "reference_answer": "dfs.replication 默认值通常为 3。",
            "category": "keyword_lookup",
            "expected_context_keywords": ["HDFS", "replication", "dfs"],
            "min_context_relevance": 0.5,
        },
        {
            "id": "rag-14",
            "query": "Zookeeper 的 clientPort 配置项是什么？",
            "reference_answer": "Zookeeper 的 clientPort 是客户端连接端口，默认值为 2181。",
            "category": "keyword_lookup",
            "expected_context_keywords": ["Zookeeper", "clientPort", "2181"],
            "min_context_relevance": 0.5,
        },
        # --- Chinese + English mixed ---
        {
            "id": "rag-15",
            "query": "How to use SDK to create a Linux 测试环境？",
            "reference_answer": "使用 SDK 的 create_environment 方法可以创建 Linux 测试环境。",
            "category": "mixed_lang",
            "expected_context_keywords": ["SDK", "create", "Linux", "环境"],
            "min_context_relevance": 0.4,
        },
    ]


def get_rag_cases() -> List[Dict[str, Any]]:
    """Return the RAG golden case list (fresh copy each call)."""
    import copy
    return copy.deepcopy(_rag_cases())


def get_cases_by_category(category: str) -> List[Dict[str, Any]]:
    """Filter cases by category."""
    return [c for c in get_rag_cases() if c["category"] == category]


def get_category_summary() -> Dict[str, int]:
    """Return case count per category."""
    cases = get_rag_cases()
    summary: Dict[str, int] = {}
    for c in cases:
        summary[c["category"]] = summary.get(c["category"], 0) + 1
    return summary
