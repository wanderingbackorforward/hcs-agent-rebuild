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
