"""Agent memory module - layered memory for multi-turn conversations.

Short-term memory: context window management with rolling summary.
Long-term memory: vector DB storage with RAG retrieval and memory gating.

Interview talking point: "My project implements layered memory — short-term
uses context window with rolling summary compression, long-term uses vector
DB with RAG retrieval, memory gating decides what to persist."
"""
from agents.memory.short_term_memory import ShortTermMemory
from agents.memory.long_term_memory import LongTermMemory

__all__ = ["ShortTermMemory", "LongTermMemory"]
