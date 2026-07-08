"""Memory service - unified facade for layered memory across agents.

Interview talking point: "All three memory layers are managed by a single
MemoryService per session. This ensures short-term, long-term, and task
memory share the same session context and don't diverge. Agents receive
the service via dependency injection rather than each creating their own."
"""
import logging

from agents.memory.short_term_memory import ShortTermMemory
from agents.memory.long_term_memory import LongTermMemory
from agents.memory.task_memory import TaskMemory

logger = logging.getLogger(__name__)


class MemoryService:
    """Unified memory facade — one instance per session_id.

    Holds STM, LTM, and TaskMemory so agents share the same memory
    instead of each instantiating their own.
    """

    def __init__(self, session_id: str, llm=None, session_repo=None,
                 embedder=None, store=None, reranker=None):
        self.session_id = session_id

        self.short_term = ShortTermMemory(llm=llm)

        self.long_term = LongTermMemory(
            llm=llm, embedder=embedder, store=store, reranker=reranker,
        )

        self.task_memory = TaskMemory(
            session_repo=session_repo, session_id=session_id,
        )

    def reset_all(self):
        """Clear all memory layers (full session reset)."""
        self.short_term.clear()
        self.task_memory.archive()
        # Long-term memory is NOT cleared — it's cross-session.

    def reset_task(self):
        """Clear only task memory (route switch within same session)."""
        self.task_memory.archive()
