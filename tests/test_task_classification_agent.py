"""Golden tests for task classification intent routing."""
import json
import os

import pytest

from agents.task_classification.task_classifier import TaskClassifier
from agents.task_classification.json_utils import parse_classification_json


# 50 条 golden test cases for intent routing
GOLDEN_TESTS = [
    # environment_match cases
    ("帮我找一个 test 环境，要有 MySQL 和 Redis", "environment_match"),
    ("推荐一个 beijing 的 staging 环境", "environment_match"),
    ("我想确认当前环境是否可用", "environment_match"),
    ("查一下上海有哪些测试环境", "environment_match"),
    ("需要 Kafka + MongoDB 的环境", "environment_match"),
    ("给我找一个 dev 环境跑用例", "environment_match"),
    ("环境 hcs-test-01 的 MySQL 端口通吗", "environment_match"),
    ("哪个环境有 elasticsearch", "environment_match"),
    ("我要验证测试环境组件是否齐全", "environment_match"),
    ("匹配一个可用的测试环境", "environment_match"),
    ("找资源充足的环境", "environment_match"),
    ("列出 beijing 所有 available 的环境", "environment_match"),
    ("我需要 MySQL 8.0 的测试环境", "environment_match"),
    ("Redis 专项测试环境在哪里", "environment_match"),
    ("确认 staging 环境状态", "environment_match"),
    ("探测 10.0.1.10 的 3306 端口", "environment_match"),
    ("环境里有没有 kafka", "environment_match"),
    ("hcs-staging-01 能不能用", "environment_match"),
    ("给我推荐一个适合跑回归测试的环境", "environment_match"),
    ("测试环境需要包含 mongodb", "environment_match"),
    ("查看 hcs-test-02 的组件", "environment_match"),
    ("找一个 region 是上海的环境", "environment_match"),
    ("环境匹配：dev + mysql + redis", "environment_match"),
    ("确认所有组件 available", "environment_match"),
    ("哪个环境有 busy 状态", "environment_match"),
    # knowledge_qa cases
    ("HCS SDK 怎么安装", "knowledge_qa"),
    ("如何初始化 HCS 客户端", "knowledge_qa"),
    ("测试环境规范要求是什么", "knowledge_qa"),
    ("HCS 部署分几个阶段", "knowledge_qa"),
    ("smoke test 是什么", "knowledge_qa"),
    ("Access Key 怎么配置", "knowledge_qa"),
    ("HCS 用户手册在哪里看", "knowledge_qa"),
    ("回归测试要做哪些事", "knowledge_qa"),
    ("HCS SDK 支持哪个 Python 版本", "knowledge_qa"),
    ("region 参数怎么填", "knowledge_qa"),
    ("MySQL 5.7 和 8.0 有什么区别", "knowledge_qa"),
    ("hcs-deploy 工具怎么用", "knowledge_qa"),
    ("验收阶段需要做什么", "knowledge_qa"),
    ("内部测试规范有什么要求", "knowledge_qa"),
    ("安装阶段要注意什么", "knowledge_qa"),
    ("准备阶段需要什么", "knowledge_qa"),
    ("HCS 混合云平台是什么", "knowledge_qa"),
    ("pip install hcs-sdk 之后怎么配置", "knowledge_qa"),
    ("网络规划在部署的哪个阶段", "knowledge_qa"),
    ("许可证是部署必须的吗", "knowledge_qa"),
    ("Redis 5.0+ 是强制要求吗", "knowledge_qa"),
    ("Kafka 版本要求是多少", "knowledge_qa"),
    ("组件必须处于 available 状态吗", "knowledge_qa"),
    ("端口不可连通怎么办", "knowledge_qa"),
    ("SDK 文档在哪里", "knowledge_qa"),
    # unrelated cases
    ("今天天气怎么样", "unrelated"),
    ("帮我点外卖", "unrelated"),
    ("讲个笑话", "unrelated"),
    ("推荐一部电影", "unrelated"),
    ("你会写 Python 吗", "unrelated"),
]


def _has_api_key() -> bool:
    return bool(os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY"))


class FakeLLM:
    """A deterministic fake LLM for golden test validation."""

    def __init__(self, intent: str):
        self._intent = intent

    async def astream(self, messages):
        payload = json.dumps({
            "intent_type": self._intent,
            "required_fields": {},
            "missing_fields": [],
            "keywords": [],
            "topic": "test"
        })
        yield type("Chunk", {"content": payload}, {})()


@pytest.mark.parametrize("user_input, expected_intent", GOLDEN_TESTS)
def test_parse_golden_json(user_input, expected_intent):
    """Verify classifier JSON parsing works for expected intent strings."""
    result = parse_classification_json(json.dumps({"intent_type": expected_intent}))
    assert result["intent_type"] == expected_intent


@pytest.mark.skipif(not _has_api_key(), reason="LLM_API_KEY not set")
@pytest.mark.asyncio
@pytest.mark.parametrize("user_input, expected_intent", GOLDEN_TESTS[:10])
async def test_real_llm_classification(user_input, expected_intent):
    from config.model_provider import create_chat_model
    classifier = TaskClassifier(create_chat_model(temperature=0))
    result = await classifier.classify(user_input)
    assert result.get("intent_type") == expected_intent
