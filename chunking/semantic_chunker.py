"""
Semantic chunker — splits text at topic boundaries using embedding similarity.

Replaces LangChain SemanticChunker with a direct implementation that:
  1. Splits text into sentences
  2. Embeds ALL sentences in ONE batched API call (fast)
  3. Detects boundaries where cosine similarity drops below threshold
  4. Groups sentences into chunks at those boundaries

This avoids the N×(N-1) API calls that LangChain's SemanticChunker made.
"""
from __future__ import annotations

import logging
import re
from typing import List, Literal

from config import settings

logger = logging.getLogger(__name__)

_MIN_CHUNK_CHARS = 200
_MAX_CHUNK_CHARS = 2000


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences using simple regex."""
    raw = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in raw if s.strip()]
    return sentences


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na  = sum(x * x for x in a) ** 0.5
    nb  = sum(x * x for x in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _embed_sync(texts: List[str], model: Literal["small", "large"]) -> List[List[float]]:
    """Synchronous embedding — called from a thread via asyncio.to_thread."""
    import httpx

    model_name = (
        settings.EMBED_MODEL_SMALL if model == "small" else settings.EMBED_MODEL_LARGE
    )
    # Batch into chunks of EMBED_BATCH_SIZE
    all_embeddings: List[List[float]] = []
    batch_size = settings.EMBED_BATCH_SIZE

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        clean = [t if t.strip() else " " for t in batch]

        response = httpx.post(
            f"{settings.OPENROUTER_BASE_URL}/embeddings",
            headers={
                "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={"model": model_name, "input": clean},
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        batch_embs = [d["embedding"] for d in sorted(data["data"], key=lambda x: x["index"])]
        all_embeddings.extend(batch_embs)

    return all_embeddings


def split_semantic(text: str, model: Literal["small", "large"] = "small") -> List[str]:
    """
    Split text into semantic chunks using cosine similarity between sentence embeddings.

    All sentences are embedded in one batched call.
    A boundary is created when similarity between adjacent sentences drops below the
    threshold (derived from the percentile config).
    """
    if not text or not text.strip():
        return []

    if len(text) < 500:
        return [text.strip()]

    sentences = _split_sentences(text)
    if len(sentences) <= 2:
        return [text.strip()]

    logger.info("[SEMANTIC_CHUNKER] Embedding %d sentences | model=%s", len(sentences), model)
    try:
        embeddings = _embed_sync(sentences, model=model)
        logger.info("[SEMANTIC_CHUNKER] Embeddings received | count=%d", len(embeddings))

        # Compute similarity between adjacent sentences
        similarities = [
            _cosine(embeddings[i], embeddings[i + 1])
            for i in range(len(embeddings) - 1)
        ]

        # Threshold = percentile of similarities
        sorted_sims = sorted(similarities)
        pct = settings.SEMANTIC_BREAKPOINT_PERCENTILE / 100.0
        idx = max(0, int(len(sorted_sims) * pct) - 1)
        threshold = sorted_sims[idx]

        # Build chunks by grouping sentences between boundaries
        chunks: List[str] = []
        current: List[str] = [sentences[0]]

        for i, sim in enumerate(similarities):
            next_sent = sentences[i + 1]
            if sim < threshold:
                chunk_text = " ".join(current).strip()
                if chunk_text:
                    chunks.append(chunk_text)
                current = [next_sent]
            else:
                current.append(next_sent)

        if current:
            chunks.append(" ".join(current).strip())

        # Merge chunks that are too short into the next one
        merged: List[str] = []
        buffer = ""
        for chunk in chunks:
            buffer = (buffer + " " + chunk).strip() if buffer else chunk
            if len(buffer) >= _MIN_CHUNK_CHARS:
                merged.append(buffer)
                buffer = ""
        if buffer:
            if merged:
                merged[-1] = (merged[-1] + " " + buffer).strip()
            else:
                merged.append(buffer)

        result = [c for c in merged if c]
        logger.debug(
            "[SEMANTIC_CHUNKER] Split | sentences=%d chunks=%d model=%s",
            len(sentences), len(result), model,
        )
        return result

    except Exception as e:
        logger.warning("[SEMANTIC_CHUNKER] Falling back to paragraph split | err=%s", e)
        paras = [p.strip() for p in text.split("\n\n") if p.strip()]
        return paras if paras else [text.strip()]
