from __future__ import annotations

import asyncio
import logging
from typing import List, Literal

from openai import AsyncOpenAI

from config import settings

logger = logging.getLogger(__name__)

_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(
            api_key=settings.OPENROUTER_API_KEY,
            base_url=settings.OPENROUTER_BASE_URL,
        )
    return _client


async def embed_texts(
    texts: List[str],
    model: Literal["small", "large"] = "small",
) -> List[List[float]]:
    """
    Embed a list of texts in batches of EMBED_BATCH_SIZE.
    model="small" → text-embedding-3-small (1536-dim)
    model="large" → text-embedding-3-large (3072-dim)
    """
    if not texts:
        return []

    model_name = (
        settings.EMBED_MODEL_SMALL if model == "small" else settings.EMBED_MODEL_LARGE
    )
    client = _get_client()
    all_embeddings: List[List[float]] = []

    for i in range(0, len(texts), settings.EMBED_BATCH_SIZE):
        batch = texts[i : i + settings.EMBED_BATCH_SIZE]
        # Strip None / empty strings — OpenRouter rejects them
        clean = [t if t and t.strip() else " " for t in batch]
        try:
            response = await client.embeddings.create(model=model_name, input=clean)
            batch_embs = [d.embedding for d in sorted(response.data, key=lambda x: x.index)]
            all_embeddings.extend(batch_embs)
            logger.debug(
                "[EMBED] batch done | model=%s offset=%d count=%d",
                model_name, i, len(batch_embs),
            )
        except Exception as e:
            logger.error("[EMBED] batch failed | model=%s offset=%d error=%s", model_name, i, e)
            dim = settings.EMBED_DIM_SMALL if model == "small" else settings.EMBED_DIM_LARGE
            all_embeddings.extend([[0.0] * dim] * len(batch))

    return all_embeddings


async def embed_query(
    query: str,
    model: Literal["small", "large"] = "small",
) -> List[float]:
    """Embed a single query string."""
    results = await embed_texts([query], model=model)
    return results[0] if results else []
