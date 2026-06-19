"""
Image captioning via OpenRouter vision LLM (no local models).

Flow:
  Image bytes → resize → base64 → OpenRouter VISION_MODEL → caption string
  Caption → text-embedding-3-small (in text_embedder.py) → 1536-dim vector
"""
from __future__ import annotations

import asyncio
import base64
import io
import logging
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

MAX_DIMENSION = 1024


def _resize_image(image_bytes: bytes, max_dim: int = MAX_DIMENSION) -> bytes:
    from PIL import Image

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = img.size
    if max(w, h) > max_dim:
        ratio = max_dim / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def save_image_locally(image_bytes: bytes, dest_dir: Path) -> str:
    """Save raw image bytes to dest_dir, return absolute path."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{uuid.uuid4().hex}.png"
    dest = dest_dir / fname
    dest.write_bytes(image_bytes)
    return str(dest)


def prepare_image_for_caption(image_path: str) -> str:
    """Read image from path, resize, return base64 data URI."""
    raw = Path(image_path).read_bytes()
    resized = _resize_image(raw)
    b64 = base64.b64encode(resized).decode("utf-8")
    return f"data:image/png;base64,{b64}"


_CAPTION_PROMPT = (
    "You are an image description assistant. "
    "Describe this image in detail covering: what is shown, any visible text, "
    "data from charts or graphs, and overall context. "
    "Be concise but complete (2-5 sentences). "
    "Focus on information that would help answer questions about this image."
)


async def generate_caption(image_path: str, openrouter_client) -> str:
    """
    Caption an image using the configured OpenRouter vision model (VISION_MODEL).
    Returns the caption string, or empty string on failure.
    """
    from config import settings

    try:
        data_uri = await asyncio.to_thread(prepare_image_for_caption, image_path)
        response = await openrouter_client.chat.completions.create(
            model=settings.VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _CAPTION_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {"url": data_uri, "detail": "low"},
                        },
                    ],
                }
            ],
            max_tokens=400,
        )
        caption = response.choices[0].message.content or ""
        logger.info(
            "[IMAGE_HANDLER] Caption generated | model=%s path=%s len=%d",
            settings.VISION_MODEL, image_path, len(caption),
        )
        return caption.strip()
    except Exception as e:
        logger.error("[IMAGE_HANDLER] Caption failed | path=%s err=%s", image_path, e)
        return ""
