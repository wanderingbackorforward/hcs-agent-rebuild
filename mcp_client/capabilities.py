"""Server capability profile and client-side feature flags.

Two layers:
  1. ServerCapabilityProfile — parsed from the MCP Server's InitializeResult.
     Reflects what the Server *actually* supports. Immutable after init.
  2. ClientFeatureFlags — sits on top of the profile, allows the Host to
     override (force-disable) any capability via env vars or runtime config.
     This is the "pure client-side switch" — if the Server doesn't support
     logging, or the Host operator wants to disable it regardless, the flag
     is False and the upper layer never sends the request.

Design principle: no glue code. The upper layer checks a boolean and skips
the request entirely. There is no fallback simulation for optional capabilities.
"""
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ServerCapabilityProfile:
    """Immutable snapshot of what the MCP Server declared during initialize.

    Each field maps to a capability in the MCP spec's ServerCapabilities.
    A capability is True only if the Server explicitly advertised it.
    """

    tools_enabled: bool = False
    resources_enabled: bool = False
    prompts_enabled: bool = False
    logging_enabled: bool = False
    completions_enabled: bool = False

    # Fine-grained sub-capabilities (only meaningful if the parent is True).
    resource_subscriptions: bool = False
    resource_list_changed: bool = False
    tool_list_changed: bool = False
    prompt_list_changed: bool = False

    # Raw ServerCapabilities for debugging / advanced use.
    raw: Optional[Dict[str, Any]] = field(default=None, repr=False)

    @classmethod
    def from_server_capabilities(cls, caps: Any) -> "ServerCapabilityProfile":
        """Build a profile from an MCP SDK ServerCapabilities object.

        ``caps`` is expected to be ``mcp.types.ServerCapabilities`` or None.
        If None (server returned nothing), all capabilities default to False.
        """
        if caps is None:
            return cls()

        raw_dict: Dict[str, Any] = {}
        try:
            # Pydantic v2 model_dump
            raw_dict = caps.model_dump(exclude_none=True) if hasattr(caps, "model_dump") else {}
        except Exception:
            raw_dict = {}

        tools = getattr(caps, "tools", None)
        resources = getattr(caps, "resources", None)
        prompts = getattr(caps, "prompts", None)
        logging_cap = getattr(caps, "logging", None)
        completions = getattr(caps, "completions", None)

        return cls(
            tools_enabled=tools is not None,
            resources_enabled=resources is not None,
            prompts_enabled=prompts is not None,
            logging_enabled=logging_cap is not None,
            completions_enabled=completions is not None,
            resource_subscriptions=getattr(resources, "subscribe", False) or False if resources else False,
            resource_list_changed=getattr(resources, "listChanged", False) or False if resources else False,
            tool_list_changed=getattr(tools, "listChanged", False) or False if tools else False,
            prompt_list_changed=getattr(prompts, "listChanged", False) or False if prompts else False,
            raw=raw_dict or None,
        )

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ServerCapabilityProfile":
        """Build a profile from a plain dict (e.g., synthesized for local server).

        Expected keys (all optional, default False):
            tools, resources, prompts, logging, completions,
            resource_subscriptions, resource_list_changed,
            tool_list_changed, prompt_list_changed
        """
        def _flag(key: str) -> bool:
            v = d.get(key)
            return bool(v) if v is not None else False

        return cls(
            tools_enabled=_flag("tools"),
            resources_enabled=_flag("resources"),
            prompts_enabled=_flag("prompts"),
            logging_enabled=_flag("logging"),
            completions_enabled=_flag("completions"),
            resource_subscriptions=_flag("resource_subscriptions"),
            resource_list_changed=_flag("resource_list_changed"),
            tool_list_changed=_flag("tool_list_changed"),
            prompt_list_changed=_flag("prompt_list_changed"),
            raw=d,
        )

    def summary(self) -> str:
        """One-line summary for logging."""
        parts = []
        if self.tools_enabled:
            parts.append("tools")
        if self.resources_enabled:
            parts.append("resources")
        if self.prompts_enabled:
            parts.append("prompts")
        if self.logging_enabled:
            parts.append("logging")
        if self.completions_enabled:
            parts.append("completions")
        return ", ".join(parts) if parts else "(none)"


@dataclass
class ClientFeatureFlags:
    """Client-side feature switches derived from ServerCapabilityProfile.

    The Host can force-disable any capability regardless of what the Server
    declared. This is the "pure client-side switch" mechanism:
    - If Server doesn't support logging → logging_enabled is False.
    - If Host operator sets MCP_DISABLE_LOGGING=true → also False.
    - Upper layer checks the flag and skips the request. No glue code.

    Environment variables (all optional, default = respect server capability):
        MCP_DISABLE_TOOLS=true       — force-disable tools
        MCP_DISABLE_RESOURCES=true   — force-disable resources
        MCP_DISABLE_PROMPTS=true     — force-disable prompts
        MCP_DISABLE_LOGGING=true     — force-disable logging
        MCP_DISABLE_COMPLETIONS=true — force-disable completions
    """

    tools_enabled: bool = True
    resources_enabled: bool = True
    prompts_enabled: bool = True
    logging_enabled: bool = True
    completions_enabled: bool = True

    @classmethod
    def from_profile(
        cls,
        profile: ServerCapabilityProfile,
        env: Optional[Dict[str, str]] = None,
    ) -> "ClientFeatureFlags":
        """Derive flags from server profile, then apply env var overrides.

        Env vars can only force-disable (never force-enable) a capability.
        This is intentional: if the Server doesn't support something, the
        Client must not pretend it does.
        """
        env = env if env is not None else dict(os.environ)

        def _disabled(var_name: str) -> bool:
            val = env.get(var_name, "")
            return val.strip().lower() in ("1", "true", "yes", "on")

        tools = profile.tools_enabled and not _disabled("MCP_DISABLE_TOOLS")
        resources = profile.resources_enabled and not _disabled("MCP_DISABLE_RESOURCES")
        prompts = profile.prompts_enabled and not _disabled("MCP_DISABLE_PROMPTS")
        logging_flag = profile.logging_enabled and not _disabled("MCP_DISABLE_LOGGING")
        completions = profile.completions_enabled and not _disabled("MCP_DISABLE_COMPLETIONS")

        if not tools:
            logger.info("MCP Client: tools disabled (server=%s, env_override=%s)",
                        profile.tools_enabled, _disabled("MCP_DISABLE_TOOLS"))
        if not resources:
            logger.info("MCP Client: resources disabled (server=%s, env_override=%s)",
                        profile.resources_enabled, _disabled("MCP_DISABLE_RESOURCES"))

        return cls(
            tools_enabled=tools,
            resources_enabled=resources,
            prompts_enabled=prompts,
            logging_enabled=logging_flag,
            completions_enabled=completions,
        )

    def to_dict(self) -> Dict[str, bool]:
        return {
            "tools_enabled": self.tools_enabled,
            "resources_enabled": self.resources_enabled,
            "prompts_enabled": self.prompts_enabled,
            "logging_enabled": self.logging_enabled,
            "completions_enabled": self.completions_enabled,
        }
