"""Repository for environments."""
from typing import List, Optional
from db.base import SessionManager
from db.models import Environment


class EnvironmentRepository:
    def __init__(self, session_manager: SessionManager):
        self._session_manager = session_manager

    def _session(self):
        return self._session_manager.get_session()

    def add(self, name: str, env_type: str, region: str = None,
            components: list = None, host: str = None, port: int = None,
            status: str = "unknown", description: str = None) -> Environment:
        with self._session() as session:
            env = Environment(
                name=name, env_type=env_type, region=region,
                components=components or [], host=host, port=port,
                status=status, description=description,
            )
            session.add(env)
            session.commit()
            session.refresh(env)
            return env

    def get_by_id(self, env_id: int) -> Optional[Environment]:
        with self._session() as session:
            return session.query(Environment).filter(
                Environment.id == env_id, Environment.is_active == 1
            ).first()

    def list_all(self) -> List[Environment]:
        with self._session() as session:
            return session.query(Environment).filter(
                Environment.is_active == 1
            ).all()

    def filter_candidates(self, env_type: str = None,
                          components: List[str] = None,
                          region: str = None) -> List[Environment]:
        with self._session() as session:
            query = session.query(Environment).filter(Environment.is_active == 1)
            if env_type:
                query = query.filter(Environment.env_type == env_type)
            if region:
                query = query.filter(Environment.region == region)
            results = query.all()
            if components:
                results = [
                    r for r in results
                    if all(c in (r.components or []) for c in components)
                ]
            return results

    def seed_defaults(self):
        defaults = [
            {
                "name": "hcs-test-01",
                "env_type": "test",
                "region": "beijing",
                "components": ["mysql", "redis", "kafka"],
                "host": "10.0.1.10",
                "port": 3306,
                "status": "available",
                "description": "标准测试环境，MySQL 8.0",
            },
            {
                "name": "hcs-test-02",
                "env_type": "test",
                "region": "shanghai",
                "components": ["mysql", "mongodb"],
                "host": "10.0.2.20",
                "port": 27017,
                "status": "busy",
                "description": "MongoDB 专项测试环境",
            },
            {
                "name": "hcs-staging-01",
                "env_type": "staging",
                "region": "beijing",
                "components": ["mysql", "redis", "elasticsearch"],
                "host": "10.0.1.50",
                "port": 3306,
                "status": "available",
                "description": "预发布环境",
            },
        ]
        with self._session() as session:
            existing = {e.name for e in session.query(Environment.name).all()}
            for d in defaults:
                if d["name"] not in existing:
                    session.add(Environment(**d))
            session.commit()
