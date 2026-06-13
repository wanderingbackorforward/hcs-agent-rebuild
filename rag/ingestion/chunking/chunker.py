"""Text chunker backed by langchain RecursiveCharacterTextSplitter.

Language-aware: zh / en separators picked at init. Default chunk_size 600
chars aligns with the 200-300 token best-range for Chinese embedding models.
"""
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter


_ZH_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
_EN_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]


class TextChunker:
    def __init__(
        self,
        chunk_size: int = 600,
        chunk_overlap: int = 100,
        lang: str = "zh",
    ):
        if lang not in ("zh", "en"):
            raise ValueError(f"Unsupported lang: {lang!r}, use 'zh' or 'en'")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.lang = lang
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=_ZH_SEPARATORS if lang == "zh" else _EN_SEPARATORS,
        )

    def split(self, text: str) -> List[str]:
        if not text:
            return []
        return self._splitter.split_text(text)
