"""Shared constants and state enums."""
from enum import Enum


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

CONFIDENCE_THRESHOLD = 0.5
