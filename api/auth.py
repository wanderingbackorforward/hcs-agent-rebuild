"""API key authentication for FastAPI endpoints.

Reads valid keys from env var `API_KEYS` (comma-separated). If empty,
the service is in dev mode and ALL requests are accepted (with a warning).
Production deployments MUST set API_KEYS.

Stores the resolved key on `request.state.api_key` so downstream deps
(rate_limit) can attribute usage without re-reading the header.
"""
import logging
import os
from typing import List

from fastapi import Header, HTTPException, Request, status

logger = logging.getLogger(__name__)


def _load_valid_keys() -> List[str]:
    raw = os.getenv("API_KEYS", "").strip()
    if not raw:
        return []
    return [k.strip() for k in raw.split(",") if k.strip()]


VALID_KEYS: List[str] = _load_valid_keys()
DEV_MODE: bool = len(VALID_KEYS) == 0

if DEV_MODE:
    logger.warning(
        "API_KEYS env var is empty — running in DEV MODE (no auth required). "
        "Set API_KEYS=key1,key2,... in production."
    )


async def require_api_key(
    request: Request,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> str:
    """FastAPI dependency: rejects requests without a valid X-API-Key.

    Stores the resolved key on request.state.api_key for downstream deps.
    """
    if DEV_MODE:
        resolved = "dev-mode-no-key"
    elif not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-API-Key header",
        )
    elif x_api_key not in VALID_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )
    else:
        resolved = x_api_key
    request.state.api_key = resolved
    return resolved
