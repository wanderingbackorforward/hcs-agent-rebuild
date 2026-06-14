"""SessionRepository tests — verifies SQLite-backed chat history persistence.

Critical: covers the SQLAlchemy JSON mutation pitfall (without
`flag_modified`, the commit is a silent no-op for in-place list/dict
changes). Each test uses a fresh temp DB so they don't interfere.
"""
import os
import tempfile

import pytest

from db.db_router import DatabaseRouter
from db.repositories.session_repository import SessionRepository


@pytest.fixture
def db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.environ["DATABASE_URL"] = f"sqlite:///{path}"
    # Force fresh import so db_config picks up the new URL.
    import importlib
    import config.database
    import db.base
    import db.db_router
    import db.repositories.session_repository
    importlib.reload(config.database)
    importlib.reload(db.base)
    importlib.reload(db.repositories.session_repository)
    importlib.reload(db.db_router)
    router = DatabaseRouter()
    yield router
    router.close()
    try:
        os.unlink(path)
    except OSError:
        pass


def test_get_or_create_creates_row(db):
    repo: SessionRepository = db.session
    s = repo.get_or_create("sess-1")
    assert s.session_id == "sess-1"
    assert s.history == [] or s.history is None
    assert s.extracted_fields == {} or s.extracted_fields is None


def test_get_or_create_returns_existing(db):
    repo: SessionRepository = db.session
    repo.get_or_create("sess-2")
    s1 = repo.get_or_create("sess-2")
    s2 = repo.get_or_create("sess-2")
    # Same row, same id
    assert s1.id == s2.id


def test_append_history_persists_across_sessions(db):
    """The core regression: appending twice should persist BOTH messages."""
    repo: SessionRepository = db.session
    repo.get_or_create("sess-3")
    repo.append_history("sess-3", "user", "hello")
    repo.append_history("sess-3", "ai", "hi back")
    history = repo.get_history("sess-3")
    assert len(history) == 2
    assert history[0] == {"role": "user", "content": "hello"}
    assert history[1] == {"role": "ai", "content": "hi back"}


def test_get_history_empty_for_unknown_session(db):
    repo: SessionRepository = db.session
    assert repo.get_history("never-existed") == []


def test_clear_history(db):
    repo: SessionRepository = db.session
    repo.get_or_create("sess-4")
    repo.append_history("sess-4", "user", "msg")
    assert len(repo.get_history("sess-4")) == 1
    repo.clear_history("sess-4")
    assert repo.get_history("sess-4") == []


def test_update_fields_persists(db):
    """Same JSON mutation pitfall as history: without flag_modified,
    update_fields would silently drop the dict.update()."""
    repo: SessionRepository = db.session
    repo.get_or_create("sess-5")
    repo.update_fields("sess-5", {"env_type": "test"})
    repo.update_fields("sess-5", {"components": ["MySQL"]})
    fields = repo.get_fields("sess-5")
    assert fields.get("env_type") == "test"
    assert fields.get("components") == ["MySQL"]


def test_get_fields_empty_for_unknown_session(db):
    repo: SessionRepository = db.session
    assert repo.get_fields("nope") == {}
