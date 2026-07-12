"""Database and Redis configuration module."""
import os
import logging

logger = logging.getLogger(__name__)


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


class RedisConfig:
    def __init__(self):
        self.url = os.getenv("REDIS_URL", "")

    @property
    def enabled(self) -> bool:
        return bool(self.url)


db_config = DatabaseConfig()
redis_config = RedisConfig()


# Module-level singleton for the Redis client.
_redis_client = None


def get_redis_client():
    """Return a cached sync Redis client, or None if Redis is not configured.

    Returns None when:
    - REDIS_URL is not set (redis_config.enabled is False)
    - The redis package is not installed
    - Client creation raises an error

    The client is created once and cached as a module-level singleton.
    Uses decode_responses=True so all string operations return Python str.
    Connection is lazy — actual connectivity is tested on first command,
    and RedisToolCache/RedisSemanticCache handle RedisError per operation.
    """
    global _redis_client
    if _redis_client is not None:
        return _redis_client
    if not redis_config.enabled:
        return None
    try:
        import redis
        _redis_client = redis.Redis.from_url(
            redis_config.url,
            decode_responses=True,
        )
        logger.info("Redis client created for url: %s", redis_config.url)
    except ImportError:
        logger.warning(
            "redis package not installed; cache falls back to in-memory"
        )
        return None
    except Exception as e:
        logger.warning("Redis client init failed: %s", e)
        return None
    return _redis_client
