"""Tests for Hybrid Search retrieval quality."""
import os
import tempfile

import pytest

from services.knowledge_service import KnowledgeService
from db.db_router import DatabaseRouter


# 30 组 query-document 命中对
RETRIEVAL_PAIRS = [
    ("HCS SDK 安装", "hcs-sdk-quickstart"),
    ("如何初始化客户端", "hcs-sdk-quickstart"),
    ("Access Key 配置", "hcs-sdk-quickstart"),
    ("pip install hcs-sdk", "hcs-sdk-quickstart"),
    ("Python 版本要求", "hcs-sdk-quickstart"),
    ("region 参数", "hcs-sdk-quickstart"),
    ("测试环境要求", "hcs-test-spec-env"),
    ("MySQL 版本要求", "hcs-test-spec-env"),
    ("Redis 版本要求", "hcs-test-spec-env"),
    ("Kafka 版本要求", "hcs-test-spec-env"),
    ("组件 available 状态", "hcs-test-spec-env"),
    ("端口可连通", "hcs-test-spec-env"),
    ("HCS 部署阶段", "hcs-manual-deploy"),
    ("准备阶段", "hcs-manual-deploy"),
    ("安装阶段", "hcs-manual-deploy"),
    ("验收阶段", "hcs-manual-deploy"),
    ("hcs-deploy 工具", "hcs-manual-deploy"),
    ("smoke test", "hcs-manual-deploy"),
    ("回归测试", "hcs-manual-deploy"),
    ("主机资源", "hcs-manual-deploy"),
    ("网络规划", "hcs-manual-deploy"),
    ("许可证", "hcs-manual-deploy"),
    ("HCS 混合云", "hcs-sdk-quickstart"),
    ("技术规范", "hcs-test-spec-env"),
    ("SDK 文档", "hcs-sdk-quickstart"),
    ("用户手册", "hcs-manual-deploy"),
    ("内部测试规范", "hcs-test-spec-env"),
    ("部署手册", "hcs-manual-deploy"),
    ("环境确认", "hcs-test-spec-env"),
    ("测试用例前置条件", "hcs-test-spec-env"),
]


def _has_api_key() -> bool:
    return bool(os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY"))


@pytest.fixture
def knowledge_service():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db_router = DatabaseRouter(db_path=f"sqlite:///{path}")
    service = KnowledgeService(db_router)
    service.initialize()
    yield service
    db_router.close()
    os.unlink(path)


def test_retrieval_hit_rate_at_10(embedding_works, knowledge_service):
    """Verify Hybrid Search top-10 hit rate >= 85%.

    Depends on the ``embedding_works`` session fixture, which live-probes the
    embedding endpoint and skips cleanly when it's unavailable (no key /
    unsupported model) rather than failing on an environment precondition.
    """
    hits = 0
    for query, expected_doc_id in RETRIEVAL_PAIRS:
        results = knowledge_service.search(query, top_k=10)
        doc_ids = {meta.get("doc_id") for _, _, _, meta in results}
        if expected_doc_id in doc_ids:
            hits += 1
    hit_rate = hits / len(RETRIEVAL_PAIRS)
    assert hit_rate >= 0.85, f"Hit rate {hit_rate:.2%} below 85%"
