"""MCP Resource: prompt_catalog — exposes available prompt template files.

This resource lists the prompt template files stored in the prompts/
directory, so the LLM (or a developer) can discover which standardized
prompts are available for reuse. It reads the filesystem at read time,
so newly added templates appear automatically.
"""
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

RESOURCE_URI = "hcs://prompts/catalog"
RESOURCE_NAME = "prompt-catalog"
RESOURCE_DESCRIPTION = (
    "Catalog of prompt template files available in the prompts/ directory. "
    "Read this to discover reusable, versioned prompt templates."
)
RESOURCE_MIME_TYPE = "text/plain"

PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts"


def prompt_catalog_handler() -> str:
    """Return a listing of all prompt template files with previews."""
    try:
        if not PROMPTS_DIR.exists():
            return "# Prompt Catalog\n\n(prompts/ directory not found)"

        files = sorted(PROMPTS_DIR.glob("*.txt"))
        if not files:
            return "# Prompt Catalog\n\n(no prompt templates found)"

        lines = ["# Prompt Catalog", ""]
        lines.append(f"**Templates found:** {len(files)}")
        lines.append("")
        for f in files:
            content = f.read_text(encoding="utf-8")
            preview = content[:100].replace("\n", " ").strip()
            lines.append(f"## {f.name}")
            lines.append(f"> {preview}...")
            lines.append("")
        return "\n".join(lines)
    except Exception as e:
        logger.warning(f"prompt_catalog resource failed: {e}")
        return f"# Prompt Catalog\n\n(Unable to load catalog: {e})"


def register_resource(protocol_handler) -> None:
    protocol_handler.register_resource(
        uri=RESOURCE_URI,
        name=RESOURCE_NAME,
        description=RESOURCE_DESCRIPTION,
        handler=prompt_catalog_handler,
        mime_type=RESOURCE_MIME_TYPE,
    )
