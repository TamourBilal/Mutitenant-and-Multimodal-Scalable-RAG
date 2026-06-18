from __future__ import annotations

from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings

# HTML-aware separators: prefer block-level breaks first
_HTML_SEPARATORS = [
    "\n\n",
    "\n",
    ". ",
    "! ",
    "? ",
    "; ",
    ", ",
    " ",
    "",
]


def split_html(text: str) -> List[str]:
    """Split clean HTML text with recursive character splitter."""
    splitter = RecursiveCharacterTextSplitter(
        separators=_HTML_SEPARATORS,
        chunk_size=settings.HTML_CHUNK_SIZE,
        chunk_overlap=settings.HTML_CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = splitter.split_text(text)
    return [c.strip() for c in chunks if c.strip()]
