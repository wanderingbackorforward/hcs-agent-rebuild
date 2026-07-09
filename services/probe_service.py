"""Mock node probe service for environment validation."""
import socket
from typing import List, Dict
from db.db_router import DatabaseRouter
from config.settings import app_settings


class ProbeService:
    def __init__(self, db_router: DatabaseRouter = None):
        self.db = db_router or DatabaseRouter()

    def probe_port(self, host: str, port: int, timeout: int = None) -> Dict:
        if timeout is None:
            timeout = app_settings.probe_timeout
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return {"reachable": True, "host": host, "port": port}
        except Exception as e:
            return {"reachable": False, "host": host, "port": port, "error": str(e)}

    def validate_environment(self, env_id: int, required_components: List[str],
                             session_id: str = None) -> Dict:
        env = self.db.environment.get_by_id(env_id)
        if not env:
            return {"valid": False, "error": "Environment not found"}

        probe_result = self.probe_port(env.host, env.port)
        actual_components = env.components or []
        matched = [c for c in required_components if c in actual_components]

        self.db.validation.add(
            environment_id=env_id,
            session_id=session_id,
            probe_result=probe_result,
            matched_components=matched,
        )

        return {
            "valid": probe_result["reachable"] and len(matched) == len(required_components),
            "environment_id": env_id,
            "probe_result": probe_result,
            "required_components": required_components,
            "matched_components": matched,
        }
