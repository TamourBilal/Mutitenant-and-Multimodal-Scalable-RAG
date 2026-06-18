from __future__ import annotations

import logging
from typing import List, Literal

from config import settings

logger = logging.getLogger(__name__)


def _build_splitter(model: Literal["small", "large"]):
    """Build a LangChain SemanticChunker backed by the correct OpenRouter embedding model."""
    from langchain_experimental.text_splitter import SemanticChunker
    from langchain_openai import OpenAIEmbeddings

    model_name = (
        settings.EMBED_MODEL_SMALL if model == "small" else settings.EMBED_MODEL_LARGE
    )
    embeddings = OpenAIEmbeddings(
        model=model_name,
        openai_api_key=settings.OPENROUTER_API_KEY,
        openai_api_base=settings.OPENROUTER_BASE_URL,
    )
    return SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=settings.SEMANTIC_BREAKPOINT_PERCENTILE,
    )


def split_semantic(text: str, model: Literal["small", "large"] = "small") -> List[str]:
    """
    Split text using LangChain SemanticChunker.
    Uses embedding similarity to detect natural topic boundaries.
    Backs off to paragraph splitting if the text is too short.
    """
    if not text or not text.strip():
        return []

    # Very short texts don't need semantic splitting
    if len(text) < 500:
        return [text.strip()]

    try:
        splitter = _build_splitter(model)
        chunks = splitter.split_text(text)
        return [c.strip() for c in chunks if c.strip()]
    except Exception as e:
        logger.warning("[SEMANTIC_CHUNKER] Falling back to paragraph split | err=%s", e)
        # Fallback: split on double newlines
        paras = [p.strip() for p in text.split("\n\n") if p.strip()]
        return paras
