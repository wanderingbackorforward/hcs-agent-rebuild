"""Shared constants and state enums."""
from enum import Enum

from config.settings import app_settings


class StateEnum(Enum):
    CLASSIFY = "classify"
    ENVIRONMENT = "environment"
    KNOWLEDGE = "knowledge"
    OTHER = "other"


class SharedState:
    def __init__(self):
        self.value = StateEnum.CLASSIFY


# ---- Classification signal words (configurable) ----
# These are used by the classification processor for fast rule-based
# checks. Move or extend here when new intent domains are added.

SWITCH_WORDS = (
    "换个话题", "换话题", "退出", "不查了", "问别的", "换个问题",
    "算了", "新建任务", "不查环境", "不查这个",
)

REFERENCE_WORDS = (
    "那个", "这个", "换成", "改成", "再加", "再来一个",
    "上面那个", "刚才", "它的", "另一个", "换一个",
)

ENV_SIGNAL = (
    "环境", "组件", "节点", "hbase", "kafka", "mysql", "redis",
    "区域", "region", "测试环境", "可用", "状态", "staging", "dev",
    "探测", "端口", "匹配", "筛选",
)

KB_SIGNAL = (
    "怎么", "是什么", "文档", "安装", "初始化", "配置说明", "规范",
    "手册", "有哪些", "区别", "在哪里", "部署阶段", "许可证",
)

MULTI_INTENT_MARKERS = (
    "顺便", "同时", "另外", "还有", "以及", "再帮我", "再问",
)

# User rejection signals — when the user denies the previous routing result.
# These indicate the routing was wrong and should be collected as negative samples.
REJECTION_WORDS = (
    "不是", "错了", "不对", "不是这个", "不是我要的", "搞错了",
    "应该是", "我要问的是", "我想查的是",
)

CONFIDENCE_THRESHOLD = app_settings.classification_confidence_threshold
