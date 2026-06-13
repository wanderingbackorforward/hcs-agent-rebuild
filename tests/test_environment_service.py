"""Unit tests for environment matching service."""
import os
import tempfile

import pytest

from db.db_router import DatabaseRouter
from services.environment_service import EnvironmentService
from services.probe_service import ProbeService


@pytest.fixture
def db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db_router = DatabaseRouter(db_path=f"sqlite:///{path}")
    yield db_router
    db_router.close()
    os.unlink(path)


def test_seed_and_match(db):
    service = EnvironmentService(db)
    service.seed()
    candidates = service.match({"env_type": "test", "components": ["mysql"]})
    assert len(candidates) >= 1
    names = {c["name"] for c in candidates}
    assert "hcs-test-01" in names


def test_match_with_components(db):
    service = EnvironmentService(db)
    service.seed()
    candidates = service.match({"env_type": "test", "components": ["mysql", "redis"]})
    assert any(c["name"] == "hcs-test-01" for c in candidates)


def test_probe_validation(db):
    service = EnvironmentService(db)
    probe = ProbeService(db)
    service.seed()
    env = db.environment.get_by_id(1)
    result = probe.validate_environment(env.id, ["mysql"], session_id="test-session")
    assert "valid" in result
    assert "probe_result" in result
