"""Context lock — minimal multi-turn context persistence.

Stores exactly 4 fields per session: locked intent, extracted params, lock
time, TTL. Persisted in SessionRepository.extracted_fields["_context_lock"]
(no DB migration). The gate *judgment* lives in ClassificationProcessor
("entry point, one place"); this module is pure data plumbing.

Lifecycle (per the balanced plan):
- new intent != locked intent  -> auto-overwrite (clear old params)
- TTL expiry                    -> auto-invalidate
- env-match task fully done     -> one manual clear (EnvironmentMatchingProcessor)
"""
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

logger = logging.getLogger(__name__)

LOCK_KEY = "_context_lock"
DEFAULT_TTL = 600  # seconds


@dataclass
class ContextLock:
    """4-field lock: intent + params + lock time + ttl."""
    intent: str = ""
    params: Dict = field(default_factory=dict)
    locked_at: float = 0.0
    ttl: int = DEFAULT_TTL

    @property
    def is_expired(self) -> bool:
        return self.locked_at > 0 and (time.time() - self.locked_at) > self.ttl

    @property
    def is_active(self) -> bool:
        return bool(self.intent) and not self.is_expired


def load_lock(repo, session_id: str) -> ContextLock:
    if not repo or not session_id:
        return ContextLock()
    try:
        data = repo.get_fields(session_id).get(LOCK_KEY)
        if not data:
            return ContextLock()
        return ContextLock(
            intent=data.get("intent", ""),
            params=data.get("params", {}) or {},
            locked_at=float(data.get("locked_at", 0.0)),
            ttl=int(data.get("ttl", DEFAULT_TTL)),
        )
    except Exception as e:
        logger.warning("load_lock failed: %s", e)
        return ContextLock()


def save_lock(repo, session_id: str, intent: str, params: Optional[Dict] = None):
    if not repo or not session_id:
        return
    try:
        repo.get_or_create(session_id)  # ensure session row exists
        repo.update_fields(session_id, {LOCK_KEY: {
            "intent": intent,
            "params": params or {},
            "locked_at": time.time(),
            "ttl": DEFAULT_TTL,
        }})
    except Exception as e:
        logger.warning("save_lock failed: %s", e)


def clear_lock(repo, session_id: str):
    if not repo or not session_id:
        return
    try:
        repo.update_fields(session_id, {LOCK_KEY: None})
    except Exception as e:
        logger.warning("clear_lock failed: %s", e)
