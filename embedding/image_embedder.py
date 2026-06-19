"""
Image embedding via caption text.

Flow:
  Image file → vision LLM caption (OpenRouter) → text-embedding-3-small → 1536-dim vector

No local HuggingFace / CLIP model is needed.
The caption is embedded with the same text-embedding-3-small used for all text chunks,
so images live in the same vector space and are searchable with normal text queries.
"""
from __future__ import annotations

import logging
from typing import List

logger = logging.getLogger(__name__)


async def embed_images(image_paths: List[str]) -> List[List[float]]:
    """
    Embed a list of image file paths by:
      1. Generating a caption with the vision LLM (already done at ingest time).
      2. Embedding the caption with text-embedding-3-small.

    At ingest time the pipeline calls generate_caption() first and stores the
    caption as chunk.content, then calls embed_texts([caption], model="small").
    This function is kept for API compatibility but delegates to text embedding.
    """
    from embedding.text_embedder import embed_texts
    from parsing.image_handler import generate_caption, prepare_image_for_caption
    from pipeline.ingest_pipeline import _get_openrouter_client

    if not image_paths:
        return []

    client = await _get_openrouter_client()
    captions: List[str] = []
    for path in image_paths:
        caption = await generate_caption(path, client)
        captions.append(caption if caption else f"Image: {path}")

    return await embed_texts(captions, model="small")


async def embed_text_clip(texts: List[str]) -> List[List[float]]:
    """
    Previously used CLIP to embed query text into image space.
    Now uses text-embedding-3-small — same space as image captions.
    """
    from embedding.text_embedder import embed_texts
    return await embed_texts(texts, model="small")
