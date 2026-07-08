"""Task memory - structured storage for current task state and intermediate results.

Interview talking point: "Task memory is the third layer — it tracks the
current task's state machine (what fields are collected, what's matched)
and stores intermediate results (retrieved docs, probe results). It's
cleared when the task completes or the route switches."

Persisted under SessionRepository.extracted_fields['_task_memory'] to avoid
DB migrations. Lifecycle: task-scoped, cleared on completion.
"""
import logging
import time

logger = logging.getLogger(__name__)

TASK_KEY = "_task_memory"


class TaskMemory:
    """Structured task-scoped memory for current task state and results."""

    def __init__(self, session_repo=None, session_id: str = None):
        self._repo = session_repo
        self._session_id = session_id
        self._state = {
            "task_type": "idle",
            "progress": {},
            "results": [],
        }
        self._load()

    def _load(self):
        if not self._repo or not self._session_id:
            return
        try:
            fields = self._repo.get_fields(self._session_id)
            stored = fields.get(TASK_KEY)
            if stored and isinstance(stored, dict):
                self._state = stored
        except Exception as e:
            logger.warning("TaskMemory load failed: %s", e)

    def _save(self):
        if not self._repo or not self._session_id:
            return
        try:
            self._repo.update_fields(self._session_id, {TASK_KEY: self._state})
        except Exception as e:
            logger.warning("TaskMemory save failed: %s", e)

    def set_task(self, task_type: str):
        self._state["task_type"] = task_type
        self._state["progress"] = {}
        self._state["results"] = []
        self._save()
        logger.info("TaskMemory: task set to '%s'", task_type)

    def update_progress(self, key: str, value):
        self._state["progress"][key] = value
        self._save()

    def add_result(self, result_type: str, content: dict):
        self._state["results"].append({
            "type": result_type,
            "content": content,
            "timestamp": time.time(),
        })
        if len(self._state["results"]) > 20:
            self._state["results"] = self._state["results"][-20:]
        self._save()

    def get_results(self, result_type: str = None) -> list:
        if result_type:
            return [r for r in self._state["results"] if r["type"] == result_type]
        return list(self._state["results"])

    def get_context(self) -> str:
        if self._state["task_type"] == "idle":
            return ""
        lines = ["[任务记忆]"]
        lines.append("当前任务: {}".format(self._state["task_type"]))
        progress = self._state.get("progress", {})
        if progress:
            lines.append("进度:")
            for k, v in progress.items():
                lines.append("  {}: {}".format(k, v))
        results = self._state.get("results", [])
        if results:
            lines.append("中间结果(近{}条):".format(min(3, len(results))))
            for r in results[-3:]:
                lines.append("  - [{}] {}".format(r["type"], r["content"]))
        return "\n".join(lines)

    def archive(self):
        """Archive task — could write summary to long-term memory. Clears after."""
        logger.info("TaskMemory: archiving task '%s'", self._state["task_type"])
        self.clear()

    def clear(self):
        self._state = {"task_type": "idle", "progress": {}, "results": []}
        self._save()

    @property
    def task_type(self) -> str:
        return self._state.get("task_type", "idle")

    @property
    def progress(self) -> dict:
        return self._state.get("progress", {})
