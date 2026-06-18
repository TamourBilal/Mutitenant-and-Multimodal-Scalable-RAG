from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import List, Union

from config import settings

logger = logging.getLogger(__name__)

_clip_model = None


def _get_clip_model():
    global _clip_model
    if _clip_model is None:
        from sentence_transformers import SentenceTransformer
        _clip_model = SentenceTransformer(settings.CLIP_MODEL)
        logger.info("CLIP model loaded | model=%s dim=%d", settings.CLIP_MODEL, settings.CLIP_DIM)
    return _clip_model


def _sync_embed_images(image_paths: List[str]) -> List[List[float]]:
    from PIL import Image

    model = _get_clip_model()
    images = []
    valid_indices: List[int] = []

    for i, path in enumerate(image_paths):
        try:
            img = Image.open(path).convert("RGB")
            images.append(img)
            valid_indices.append(i)
        except Exception as e:
            logger.warning("[CLIP] Failed to open image | path=%s error=%s", path, e)

    if not images:
        return [[0.0] * settings.CLIP_DIM] * len(image_paths)

    embeddings_raw = model.encode(images, convert_to_numpy=True)

    results = [[0.0] * settings.CLIP_DIM] * len(image_paths)
    for idx, emb in zip(valid_indices, embeddings_raw):
        results[idx] = emb.tolist()

    return results


async def embed_images(image_paths: List[str]) -> List[List[float]]:
    """Encode images with CLIP (clip-ViT-B-32) → 512-dim vectors."""
    if not image_paths:
        return []
    return await asyncio.to_thread(_sync_embed_images, image_paths)


def _sync_embed_text_clip(texts: List[str]) -> List[List[float]]:
    """Embed query text with CLIP so it lands in the same 512-dim space as images."""
    model = _get_clip_model()
    embeddings = model.encode(texts, convert_to_numpy=True)
    return [e.tolist() for e in embeddings]


async def embed_text_clip(texts: List[str]) -> List[List[float]]:
    """Text→image space embedding (for text-to-image search at query time)."""
    if not texts:
        return []
    return await asyncio.to_thread(_sync_embed_text_clip, texts)
