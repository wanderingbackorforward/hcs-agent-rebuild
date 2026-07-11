"""MCP Client module — capability-aware client abstraction.

Provides a unified MCPClientBase interface with two implementations:
  - LocalMCPClient: wraps an in-process ProtocolHandler (our own server).
  - RemoteMCPClient: connects to a third-party MCP Server via stdio/SSE.

Key design: the Client owns capability negotiation. After initialize(),
the Client caches a ServerCapabilityProfile and derives ClientFeatureFlags.
Upper layers (KnowledgeToolBroker, KnowledgeQAAgent) query flags before
making requests — no glue code, just boolean switches.

Interview talking point: "My MCP Client does real capability negotiation
during the initialize handshake. If a third-party Server doesn't support
logging or resources, I turn off the local feature switch and the upper
layer never sends those requests. No glue code, pure client-side control."
"""
from .base import MCPClientBase
from .capabilities import ServerCapabilityProfile, ClientFeatureFlags
from .local_client import LocalMCPClient
from .remote_client import RemoteMCPClient

__all__ = [
    "MCPClientBase",
    "ServerCapabilityProfile",
    "ClientFeatureFlags",
    "LocalMCPClient",
    "RemoteMCPClient",
]
