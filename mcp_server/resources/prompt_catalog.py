"""MCP Resource: prompt_catalog — exposes available prompt template files.

This resource lists the prompt template files stored in the prompts/
directory, so the LLM (or a developer) can discover which standardized
prompts are available for reuse. It reads the filesystem at read time,
so newly added templates appear automatically.
"""
import logging

from prompts.loader import list_prompt_names, load_prompt
from config.audit import sanitize_text

logger = logging.getLogger(__name__)

RESOURCE_URI = "hcs://prompts/catalog"
RESOURCE_NAME = "prompt-catalog"
RESOURCE_DESCRIPTION = (
    "Catalog of prompt template files available in the prompts/ directory. "
    "Read this to discover reusable, versioned prompt templates."
)
RESOURCE_MIME_TYPE = "text/plain"


def prompt_catalog_handler() -> str:
    """Return a listing of all prompt template files with previews."""
    try:
        names = list_prompt_names()
        if not names:
            return "# Prompt Catalog\n\n(no prompt templates found)"

        lines = ["# Prompt Catalog", ""]
        lines.append(f"**Templates found:** {len(names)}")
        lines.append("")
        for name in names:
            content = load_prompt(name)
            preview = content[:100].replace("\n", " ").strip()
            lines.append(f"## {name}")
            lines.append(f"> {preview}...")
            lines.append("")
        return "\n".join(lines)
    except Exception as e:
        safe_msg = sanitize_text(str(e))
        logger.warning(f"prompt_catalog resource failed: {safe_msg}")
        return f"# Prompt Catalog\n\n(Unable to load catalog: {safe_msg})"


def register_resource(protocol_handler) -> None:
    protocol_handler.register_resource(
        uri=RESOURCE_URI,
        name=RESOURCE_NAME,
        description=RESOURCE_DESCRIPTION,
        handler=prompt_catalog_handler,
        mime_type=RESOURCE_MIME_TYPE,
    )
