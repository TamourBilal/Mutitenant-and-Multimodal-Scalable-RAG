from __future__ import annotations

from typing import List

from config import settings

# Prefer block-level breaks first, fall back to word/char splits
_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]


def _split_on_separator(text: str, separator: str, chunk_size: int, overlap: int) -> List[str]:
    """Recursively split text using the separator list."""
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    if not separator:
        # Hard split at chunk_size
        chunks = []
        for i in range(0, len(text), chunk_size - overlap):
            chunks.append(text[i : i + chunk_size])
        return chunks

    parts = text.split(separator)
    chunks: List[str] = []
    current = ""

    for part in parts:
        candidate = (current + separator + part).strip() if current else part.strip()
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            # Part itself too big — recurse with next separator
            if len(part) > chunk_size:
                next_sep_idx = _SEPARATORS.index(separator) + 1 if separator in _SEPARATORS else len(_SEPARATORS)
                next_sep = _SEPARATORS[next_sep_idx] if next_sep_idx < len(_SEPARATORS) else ""
                chunks.extend(_split_on_separator(part, next_sep, chunk_size, overlap))
                current = ""
            else:
                current = part.strip()

    if current:
        chunks.append(current)

    # Apply overlap: prepend tail of previous chunk
    if overlap > 0 and len(chunks) > 1:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            tail = overlapped[-1][-overlap:]
            overlapped.append((tail + " " + chunks[i]).strip())
        return overlapped

    return chunks


def split_html(text: str) -> List[str]:
    """Split clean HTML/plain text with recursive character splitter. No external deps."""
    if not text or not text.strip():
        return []
    chunks = _split_on_separator(
        text.strip(),
        separator=_SEPARATORS[0],
        chunk_size=settings.HTML_CHUNK_SIZE,
        overlap=settings.HTML_CHUNK_OVERLAP,
    )
    return [c for c in chunks if c.strip()]
