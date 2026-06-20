"""
Recursive chunker — splits text into size-bounded, overlapping windows.

Replaces the old embedding-based semantic chunker. Embedding-similarity
boundary detection shredded structured/tabular pages (it broke at abbreviation
periods like 'const.' / 'Pop.' and disconnected label/value rows) and made an
embedding API call per document just to chunk.

This implementation:
  1. Packs whole lines together using a recursive separator list
     (paragraph → line → sentence → word → char), never breaking mid-word
     unless a single token exceeds the window.
  2. Bounds every chunk to <= _MAX_CHUNK_CHARS with _OVERLAP carry-over.
  3. Merges chunks below _MIN_CHUNK_CHARS into their neighbour so no tiny,
     context-free fragments survive.

No external API calls — fast and deterministic. Loses no content.
"""
from __future__ import annotations

import logging
from typing import List, Literal

logger = logging.getLogger(__name__)

_MIN_CHUNK_CHARS = 400
_MAX_CHUNK_CHARS = 3000
_OVERLAP = 300


def _merge_small(chunks: List[str], min_chars: int = _MIN_CHUNK_CHARS) -> List[str]:
    """Merge chunks shorter than `min_chars` into the previous (or next) chunk."""
    merged: List[str] = []
    for c in chunks:
        c = c.strip()
        if not c:
            continue
        if merged and len(merged[-1]) < min_chars:
            merged[-1] = (merged[-1] + " " + c).strip()
        else:
            merged.append(c)
    # If the final chunk is still too small, fold it back into the previous one.
    if len(merged) > 1 and len(merged[-1]) < min_chars:
        merged[-2] = (merged[-2] + " " + merged.pop()).strip()
    return merged


def split_recursive(
    text: str,
    size: int = _MAX_CHUNK_CHARS,
    overlap: int = _OVERLAP,
    min_chars: int = _MIN_CHUNK_CHARS,
) -> List[str]:
    """
    Line-aware recursive character packing into <=`size` windows with `overlap`,
    then merge fragments below `min_chars`. Loses no content.
    """
    if not text or not text.strip():
        return []
    if len(text) <= size:
        return [text.strip()]

    from chunking.recursive_html import _split_on_separator, _SEPARATORS

    raw = _split_on_separator(text.strip(), separator=_SEPARATORS[0], chunk_size=size, overlap=overlap)
    chunks = _merge_small([c for c in raw if c.strip()], min_chars)
    logger.debug("[RECURSIVE_CHUNKER] Split | chars=%d chunks=%d size=%d overlap=%d",
                 len(text), len(chunks), size, overlap)
    return chunks


def split_semantic(text: str, model: Literal["small", "large"] = "small") -> List[str]:
    """
    Backwards-compatible entry point. Now delegates to recursive packing
    (the `model` argument is ignored — kept so callers don't need changes).
    """
    return split_recursive(text)
