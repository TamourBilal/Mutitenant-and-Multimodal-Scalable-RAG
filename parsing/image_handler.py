from __future__ import annotations

import asyncio
import base64
import io
import logging
import uuid
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

MAX_DIMENSION = 1024   # resize longest side to this before sending to OpenRouter


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


async def generate_caption(image_path: str, openrouter_client) -> str:
    """Call OpenRouter vision model to caption an image."""
    try:
        data_uri = await asyncio.to_thread(prepare_image_for_caption, image_path)
        response = await openrouter_client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Describe this image in detail. Include: what is shown, "
                                "any text visible, charts/graphs data if present, and context. "
                                "Be concise but complete (2-4 sentences)."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": data_uri, "detail": "low"},
                        },
                    ],
                }
            ],
            max_tokens=300,
        )
        caption = response.choices[0].message.content or ""
        return caption.strip()
    except Exception as e:
        logger.error("[IMAGE_HANDLER] Caption failed | path=%s err=%s", image_path, e)
        return ""
