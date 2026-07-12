"""Centralized application configuration via environment variables.

All tunable parameters live here so hard-coded magic numbers don't scatter
across the codebase. Each module imports what it needs from ``app_settings``.

Environment variables are documented in ``.env.example``.
"""
import os

from dotenv import load_dotenv

load_dotenv()


def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    val = os.getenv(name)
    try:
        return int(val) if val not in (None, "") else default
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    val = os.getenv(name)
    try:
        return float(val) if val not in (None, "") else default
    except (TypeError, ValueError):
        return default


def _env_str(name: str, default: str = "") -> str:
    val = os.getenv(name)
    return val.strip() if val and val.strip() else default


class AppSettings:
    """Centralized application configuration.

    Every attribute reads from an environment variable at import time,
    falling back to a safe default. Update ``.env`` (or export env vars)
    to override at deploy time — no code changes required.
    """

    # ---- Application metadata ----
    app_name: str = _env_str("APP_NAME", "HCS 测试辅助 Agent 平台")
    app_description: str = _env_str(
        "APP_DESCRIPTION", "HCS 测试环境匹配 + MCP 知识检索 Agent 平台"
    )
    app_version: str = _env_str("APP_VERSION", "0.1.0")
    debug: bool = _env_bool("DEBUG", True)
    log_level: str = _env_str("LOG_LEVEL", "INFO").upper()

    # ---- Server ----
    host: str = _env_str("HOST", "127.0.0.1")
    port: int = _env_int("PORT", 8000)

    # ---- CORS ----
    cors_origins: str = _env_str("CORS_ORIGINS", "*")
    cors_allow_credentials: bool = _env_bool("CORS_ALLOW_CREDENTIALS", False)

    # ---- Directories ----
    static_dir: str = _env_str("STATIC_DIR", "web/static")
    templates_dir: str = _env_str("TEMPLATES_DIR", "web/templates")
    prompts_dir: str = _env_str("PROMPTS_DIR", "prompts")

    # ---- Session cache (memory-leak prevention) ----
    session_cache_max_size: int = _env_int("SESSION_CACHE_MAX_SIZE", 100)
    session_cache_ttl: int = _env_int("SESSION_CACHE_TTL", 3600)

    # ---- Vector store ----
    chroma_persist_dir: str = _env_str("CHROMA_PERSIST_DIR", "./data/chroma")
    chroma_distance: str = _env_str("CHROMA_DISTANCE", "cosine")
    knowledge_collection: str = _env_str("KNOWLEDGE_COLLECTION", "hcs_knowledge")

    # ---- Raw document archive (Tier 2: trustworthy source for index rebuild) ----
    raw_doc_archive_dir: str = _env_str("RAW_DOC_ARCHIVE_DIR", "./data/raw_docs")

    # ---- Retrieval ----
    retrieval_top_k: int = _env_int("RETRIEVAL_TOP_K", 5)
    memory_top_k: int = _env_int("MEMORY_TOP_K", 3)

    # ---- LLM ----
    llm_temperature: float = _env_float("LLM_TEMPERATURE", 0.0)
    tiktoken_model: str = _env_str("TIKTOKEN_MODEL", "gpt-4")

    # ---- Context manager ----
    context_max_tokens: int = _env_int("CONTEXT_MAX_TOKENS", 6000)
    context_system_budget: int = _env_int("CONTEXT_SYSTEM_BUDGET", 500)
    context_response_budget: int = _env_int("CONTEXT_RESPONSE_BUDGET", 1000)

    # ---- Memory ----
    stm_max_turns: int = _env_int("STM_MAX_TURNS", 6)
    stm_keep_recent: int = _env_int("STM_KEEP_RECENT", 4)
    ltm_importance_threshold: float = _env_float("LTM_IMPORTANCE_THRESHOLD", 0.7)
    ltm_confidence_threshold: float = _env_float("LTM_CONFIDENCE_THRESHOLD", 0.15)
    ltm_recency_halflife: int = _env_int("LTM_RECENCY_HALFLIFE", 604800)
    ltm_ttl: int = _env_int("LTM_TTL", 2592000)
    context_lock_ttl: int = _env_int("CONTEXT_LOCK_TTL", 600)

    # ---- ReAct ----
    react_max_iterations: int = _env_int("REACT_MAX_ITERATIONS", 5)

    # ---- Chunk quality guard ----
    # Rule-based pre-filter before embedding. Off by default; enable for
    # large-document ingestion to avoid wasting embedding calls on clean chunks.
    chunk_guard_enabled: bool = _env_bool("CHUNK_GUARD_ENABLED", False)
    chunk_guard_min_chars: int = _env_int("CHUNK_GUARD_MIN_CHARS", 300)
    chunk_guard_min_separators: int = _env_int("CHUNK_GUARD_MIN_SEPARATORS", 2)
    chunk_guard_resplit_threshold: float = _env_float(
        "CHUNK_GUARD_RESPLIT_THRESHOLD", 0.45
    )

    # ---- Cache TTLs ----
    llm_cache_ttl: int = _env_int("LLM_CACHE_TTL", 1800)
    tool_cache_ttl: int = _env_int("TOOL_CACHE_TTL", 600)
    semantic_cache_ttl: int = _env_int("SEMANTIC_CACHE_TTL", 900)
    semantic_cache_threshold: float = _env_float("SEMANTIC_CACHE_THRESHOLD", 0.92)
    semantic_cache_max_entries: int = _env_int("SEMANTIC_CACHE_MAX_ENTRIES", 5000)

    # ---- Thresholds ----
    classification_confidence_threshold: float = _env_float(
        "CLASSIFICATION_CONFIDENCE_THRESHOLD", 0.5
    )
    semantic_similarity_threshold: float = _env_float(
        "SEMANTIC_SIMILARITY_THRESHOLD", 0.65
    )

    # ---- NLI (Natural Language Inference) confidence gate ----
    # When True and an embedder is available, NLIValidator checks whether
    # the user query matches the selected agent's responsibility after LLM
    # routing.  Set to False to disable (falls back to v1 LLM-only logic).
    enable_nli: bool = _env_bool("ENABLE_NLI", True)
    nli_pass_threshold: float = _env_float("NLI_PASS_THRESHOLD", 0.7)
    nli_borderline_threshold: float = _env_float("NLI_BORDERLINE_THRESHOLD", 0.6)
    # Degradation fallback: LLM self-confidence must be >= this when NLI is off.
    nli_fallback_confidence: float = _env_float("NLI_FALLBACK_CONFIDENCE", 0.8)

    # ---- Code review ----
    code_review_chunk_size: int = _env_int("CODE_REVIEW_CHUNK_SIZE", 3000)
    code_review_chunk_overlap: int = _env_int("CODE_REVIEW_CHUNK_OVERLAP", 200)

    # ---- Probe ----
    probe_timeout: int = _env_int("PROBE_TIMEOUT", 2)

    # ---- MCP server ----
    mcp_server_name: str = _env_str("MCP_SERVER_NAME", "hcs-agent-rebuild")

    # ---- MCP Client ----
    # Transport: "local" (in-process), "stdio" (subprocess), "sse" (remote).
    mcp_client_transport: str = _env_str("MCP_CLIENT_TRANSPORT", "local")
    # For stdio transport: command and args to spawn the MCP Server process.
    mcp_client_command: str = _env_str("MCP_CLIENT_COMMAND", "")
    mcp_client_args: str = _env_str("MCP_CLIENT_ARGS", "")
    # For SSE transport: URL of the remote MCP Server.
    mcp_client_url: str = _env_str("MCP_CLIENT_URL", "")
    mcp_client_timeout: float = _env_float("MCP_CLIENT_TIMEOUT", 5.0)
    # Force-disable capabilities (overrides server advertisement).
    # These are read by ClientFeatureFlags.from_profile() via os.environ.

    @property
    def cors_origins_list(self) -> list:
        """Parse CORS_ORIGINS into a list; ``*`` stays as ``['*']``."""
        if self.cors_origins.strip() == "*":
            return ["*"]
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


app_settings = AppSettings()
