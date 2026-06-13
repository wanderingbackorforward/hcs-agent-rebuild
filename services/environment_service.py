"""Environment matching service."""
from typing import List, Dict
from db.db_router import DatabaseRouter


class EnvironmentService:
    def __init__(self, db_router: DatabaseRouter = None):
        self.db = db_router or DatabaseRouter()

    def seed(self):
        self.db.environment.seed_defaults()

    def match(self, requirements: Dict) -> List[Dict]:
        env_type = requirements.get("env_type")
        components = requirements.get("components")
        region = requirements.get("region")
        candidates = self.db.environment.filter_candidates(
            env_type=env_type, components=components, region=region
        )
        return [
            {
                "id": c.id,
                "name": c.name,
                "env_type": c.env_type,
                "region": c.region,
                "components": c.components,
                "host": c.host,
                "port": c.port,
                "status": c.status,
                "description": c.description,
            }
            for c in candidates
        ]

    def get_by_id(self, env_id: int) -> Dict:
        env = self.db.environment.get_by_id(env_id)
        if not env:
            return {}
        return {
            "id": env.id,
            "name": env.name,
            "env_type": env.env_type,
            "region": env.region,
            "components": env.components,
            "host": env.host,
            "port": env.port,
            "status": env.status,
            "description": env.description,
        }
