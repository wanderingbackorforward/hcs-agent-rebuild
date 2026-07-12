"""Raw document archive — trustworthy source for index rebuild.

Stores original document content on the local filesystem, keyed by doc_id.
When the embedding model changes or chunk strategy is adjusted, the rebuild
script reads from here to re-chunk and re-vectorize without needing the
original source files.

Layout: {archive_dir}/{doc_id}/content.txt
        {archive_dir}/{doc_id}/meta.json   (category, title, source, archived_at)

For single-container deployments this is sufficient. When scaling to
multi-node, swap LocalArchive for an S3/MinIO-backed implementation with
the same interface.
"""
import json
import logging
import os
import time
from typing import Optional, Dict, List

from config.settings import app_settings

logger = logging.getLogger(__name__)


class RawDocArchive:
    """Filesystem-backed raw document archive."""

    def __init__(self, archive_dir: str = None):
        self.archive_dir = archive_dir or app_settings.raw_doc_archive_dir
        os.makedirs(self.archive_dir, exist_ok=True)

    def _doc_dir(self, doc_id: str) -> str:
        return os.path.join(self.archive_dir, doc_id)

    def archive(self, doc_id: str, content: str, category: str = "spec",
                title: str = None, source: str = None) -> str:
        """Archive raw document content. Returns the archive path.

        Overwrites if doc_id already archived (latest content wins).
        """
        doc_dir = self._doc_dir(doc_id)
        os.makedirs(doc_dir, exist_ok=True)

        content_path = os.path.join(doc_dir, "content.txt")
        with open(content_path, "w", encoding="utf-8") as f:
            f.write(content)

        meta = {
            "doc_id": doc_id,
            "category": category,
            "title": title or doc_id,
            "source": source or "inline",
            "archived_at": time.time(),
        }
        meta_path = os.path.join(doc_dir, "meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        logger.info("Archived raw doc: %s -> %s", doc_id, content_path)
        return content_path

    def read(self, doc_id: str) -> Optional[str]:
        """Read archived raw content for a doc_id."""
        content_path = os.path.join(self._doc_dir(doc_id), "content.txt")
        if not os.path.exists(content_path):
            return None
        with open(content_path, "r", encoding="utf-8") as f:
            return f.read()

    def read_meta(self, doc_id: str) -> Optional[Dict]:
        """Read archived metadata for a doc_id."""
        meta_path = os.path.join(self._doc_dir(doc_id), "meta.json")
        if not os.path.exists(meta_path):
            return None
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def delete(self, doc_id: str):
        """Remove an archived document."""
        doc_dir = self._doc_dir(doc_id)
        if os.path.exists(doc_dir):
            import shutil
            shutil.rmtree(doc_dir)
            logger.info("Deleted archived doc: %s", doc_id)

    def list_archived(self) -> List[str]:
        """List all archived doc_ids."""
        if not os.path.exists(self.archive_dir):
            return []
        return sorted(
            d for d in os.listdir(self.archive_dir)
            if os.path.isdir(os.path.join(self.archive_dir, d))
        )
