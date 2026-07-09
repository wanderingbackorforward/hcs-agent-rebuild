"""Context manager - token counting, rolling summary, and context assembly.

Interview talking point: "I implemented a context manager that uses tiktoken
for precise token counting. Context assembly is layered: system prompt,
long-term memory, summary, recent conversation."
"""
import logging

from config.settings import app_settings

logger = logging.getLogger(__name__)

try:
    import tiktoken
    _ENCODER = tiktoken.encoding_for_model(app_settings.tiktoken_model)
    _HAS_TIKTOKEN = True
except Exception:
    _HAS_TIKTOKEN = False


def count_tokens(text: str) -> int:
    if not text:
        return 0
    if _HAS_TIKTOKEN:
        return len(_ENCODER.encode(text))
    return len(text) // 2


class ContextManager:
    """Manages context window budget with token counting and overflow handling."""

    def __init__(self, max_tokens: int = None,
                 short_term_memory=None, long_term_memory=None,
                 task_memory=None):
        self.max_tokens = max_tokens or app_settings.context_max_tokens
        self.short_term = short_term_memory
        self.long_term = long_term_memory
        self.task_memory = task_memory

    def assemble_context(self, system_prompt: str, query: str,
                         max_tokens: int = None) -> dict:
        budget = max_tokens or self.max_tokens
        available = (budget - app_settings.context_system_budget
                     - app_settings.context_response_budget)
        overflow = False

        memory_context = ""
        if self.long_term:
            memory_context = self.long_term.get_context(
                query, top_k=app_settings.memory_top_k)

        task_context = ""
        if self.task_memory:
            task_context = self.task_memory.get_context()

        conversation_context = ""
        if self.short_term:
            conversation_context = self.short_term.get_context()

        total = (count_tokens(system_prompt) + count_tokens(memory_context)
                 + count_tokens(task_context)
                 + count_tokens(conversation_context) + count_tokens(query))

        if total > available:
            overflow = True
            if count_tokens(memory_context) > 200 and self.long_term:
                memory_context = self.long_term.get_context(query, top_k=1)
                total = (count_tokens(system_prompt) + count_tokens(memory_context)
                         + count_tokens(task_context)
                         + count_tokens(conversation_context) + count_tokens(query))

            if total > available and self.task_memory:
                task_context = ""  # drop task context first (least critical)
                total = (count_tokens(system_prompt) + count_tokens(memory_context)
                         + count_tokens(conversation_context) + count_tokens(query))

            if total > available and self.short_term:
                while len(self.short_term._messages) > 2 and total > available:
                    prev_len = len(self.short_term._messages)
                    self.short_term._compress()
                    if len(self.short_term._messages) >= prev_len:
                        break  # can't compress further
                    conversation_context = self.short_term.get_context()
                    total = (count_tokens(system_prompt) + count_tokens(memory_context)
                             + count_tokens(conversation_context) + count_tokens(query))

            if total > available:
                conversation_context = "[近期对话已截断]"
                total = (count_tokens(system_prompt) + count_tokens(memory_context)
                         + count_tokens(conversation_context) + count_tokens(query))

        return {
            "system_prompt": system_prompt,
            "memory_context": memory_context,
            "task_context": task_context,
            "conversation_context": conversation_context,
            "query": query,
            "token_count": total,
            "overflow": overflow,
        }

    def build_prompt(self, system_prompt: str, query: str,
                     max_tokens: int = None) -> str:
        ctx = self.assemble_context(system_prompt, query, max_tokens)
        parts = [ctx["system_prompt"]]
        if ctx["memory_context"]:
            parts.append(ctx["memory_context"])
        if ctx["task_context"]:
            parts.append(ctx["task_context"])
        if ctx["conversation_context"]:
            parts.append(ctx["conversation_context"])
        parts.append("## 当前问题\n{}".format(query))
        return "\n\n".join(parts)
