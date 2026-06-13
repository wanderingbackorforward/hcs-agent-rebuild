"""Database configuration module."""
import os


class DatabaseConfig:
    def __init__(self):
        self.db_path = os.getenv("DATABASE_URL", "sqlite:///./data/hcs_agent.db")
        self.echo = os.getenv("DB_ECHO", "false").lower() == "true"
        self.pool_size = int(os.getenv("DB_POOL_SIZE", "10"))
        self.max_overflow = int(os.getenv("DB_MAX_OVERFLOW", "20"))

    @property
    def connection_string(self) -> str:
        return self.db_path

    @property
    def is_sqlite(self) -> bool:
        return self.db_path.startswith("sqlite")

    def get_engine_kwargs(self) -> dict:
        kwargs = {"echo": self.echo}
        if not self.is_sqlite:
            kwargs.update({
                "pool_size": self.pool_size,
                "max_overflow": self.max_overflow,
            })
        return kwargs


db_config = DatabaseConfig()
