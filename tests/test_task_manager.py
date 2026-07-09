"""Tests for task manager — cooperative cancellation and checkpointing."""
import asyncio
import time
import pytest

from api.task_manager import TaskManager, get_task_manager, make_task_id


class TestMakeTaskId:
    def test_format(self):
        tid = make_task_id("sess123")
        assert tid.startswith("sess123:")
        # Hex suffix after colon.
        suffix = tid.split(":")[1]
        int(suffix, 16)  # should not raise

    def test_unique_over_time(self):
        tid1 = make_task_id("s1")
        time.sleep(0.01)
        tid2 = make_task_id("s1")
        assert tid1 != tid2


class TestTaskManager:
    def test_register_and_cancel(self):
        tm = TaskManager()
        ev = tm.register("t1")
        assert isinstance(ev, asyncio.Event)
        assert not tm.is_cancelled("t1")
        assert tm.cancel("t1") is True
        assert tm.is_cancelled("t1") is True

    def test_cancel_unknown(self):
        tm = TaskManager()
        assert tm.cancel("nonexistent") is False
        assert tm.is_cancelled("nonexistent") is False

    def test_checkpoint_save_get(self):
        tm = TaskManager()
        tm.register("t2")
        tm.checkpoint("t2", {"stage": "pre_route", "intent": "knowledge_qa"})
        cp = tm.get_checkpoint("t2")
        assert cp is not None
        assert cp["stage"] == "pre_route"
        assert cp["intent"] == "knowledge_qa"

    def test_checkpoint_missing(self):
        tm = TaskManager()
        assert tm.get_checkpoint("nope") is None

    def test_cleanup_removes_all(self):
        tm = TaskManager()
        tm.register("t3")
        tm.checkpoint("t3", {"s": 1})
        tm.cleanup("t3")
        assert not tm.is_cancelled("t3")
        assert tm.get_checkpoint("t3") is None

    def test_prune_removes_expired(self):
        tm = TaskManager(ttl=0.01)  # 10ms TTL
        tm.register("t4")
        tm.checkpoint("t4", {"s": 1})
        time.sleep(0.02)
        removed = tm.prune()
        assert removed == 1
        assert tm.get_checkpoint("t4") is None

    def test_prune_keeps_active(self):
        tm = TaskManager(ttl=300)
        tm.register("t5")
        tm.checkpoint("t5", {"s": 1})
        removed = tm.prune()
        assert removed == 0
        assert tm.get_checkpoint("t5") is not None

    def test_cancelled_event_can_be_checked_sync(self):
        """asyncio.Event.is_set() is safe to call from any context."""
        tm = TaskManager()
        tm.register("t6")
        ev = tm._events["t6"]
        assert not ev.is_set()
        tm.cancel("t6")
        assert ev.is_set()

    def test_get_task_manager_singleton(self):
        tm1 = get_task_manager()
        tm2 = get_task_manager()
        assert tm1 is tm2
