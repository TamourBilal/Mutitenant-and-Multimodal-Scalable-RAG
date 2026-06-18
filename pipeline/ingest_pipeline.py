"""
Full ingestion pipeline:
  intent detection → parse → chunk → embed → upsert to Weaviate → update SQLite

The intent layer (intent/detector.py) runs first on every document and determines:
  - doc_type (legal/news/medical/financial/research/pdf/html/image/other)
  - chunking_strategy (semantic/recursive/image_caption)
  - embed_model (small/large/clip)
  - named_vector (text_small/text_large/image)
  - special flags: is_legal, is_table_heavy, is_scanned, is_mixed_content

Date extraction:
  Each parsed document is scanned for a publication / creation date.
  Extracted date → stored as doc_date (source="extracted").
  No date found    → ingestion timestamp (source="ingestion").

Local metadata:
  A JSON file is written to storage/metadata/{user_id}/{doc_id}.json
  immediately after ingestion for fast lookup without DB queries.
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import settings
from chunking.router import Chunk, chunk_image_caption, route_and_chunk
from embedding.image_embedder import embed_images
from embedding.text_embedder import embed_texts
from intent.detector import DocumentIntent, detect_intent
from parsing.date_extractor import (
    extract_date_from_html,
    extract_date_from_pdf_text,
    extract_date_from_text,
)
from parsing.html_parser import parse_html
from parsing.image_handler import generate_caption, save_image_locally
from parsing.pdf_parser import parse_pdf
from weaviate_store.client import ensure_tenant
from weaviate_store.ingestor import WeaviateObject, batch_upsert

logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _get_openrouter_client():
    from openai import AsyncOpenAI
    return AsyncOpenAI(
        api_key=settings.OPENROUTER_API_KEY,
        base_url=settings.OPENROUTER_BASE_URL,
    )


def _chunk_to_props(chunk: Chunk, doc_date_iso: str) -> Dict[str, Any]:
    return {
        "content":     chunk.content,
        "source":      chunk.source,
        "doc_type":    chunk.doc_type,
        "date":        doc_date_iso,   # publication date (or ingestion if not found)
        "chunk_index": chunk.chunk_index,
        "page_no":     chunk.page_no,
        "file_path":   chunk.file_path,
        "chunk_type":  chunk.chunk_type,
    }


async def _embed_and_build_objects(
    chunks: List[Chunk], doc_date_iso: str
) -> List[WeaviateObject]:
    """Embed chunks grouped by model, build WeaviateObject list."""
    if not chunks:
        return []

    small_idx = [i for i, c in enumerate(chunks) if c.embed_model == "small"]
    large_idx = [i for i, c in enumerate(chunks) if c.embed_model == "large"]
    clip_idx  = [i for i, c in enumerate(chunks) if c.embed_model == "clip"]

    tasks, keys = [], []
    if small_idx:
        tasks.append(embed_texts([chunks[i].content for i in small_idx], model="small"))
        keys.append(("small", small_idx))
    if large_idx:
        tasks.append(embed_texts([chunks[i].content for i in large_idx], model="large"))
        keys.append(("large", large_idx))
    if clip_idx:
        tasks.append(embed_images([chunks[i].file_path for i in clip_idx]))
        keys.append(("clip", clip_idx))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    emb_map: Dict[int, tuple] = {}
    for (model_key, indices), result in zip(keys, results):
        if isinstance(result, Exception):
            logger.error("[PIPELINE] Embed failed | model=%s err=%s", model_key, result)
            dim = (settings.EMBED_DIM_LARGE if model_key == "large"
                   else settings.CLIP_DIM if model_key == "clip"
                   else settings.EMBED_DIM_SMALL)
            result = [[0.0] * dim] * len(indices)
        for chunk_i, emb in zip(indices, result):
            emb_map[chunk_i] = (chunks[chunk_i].named_vector, emb)

    objects: List[WeaviateObject] = []
    for i, chunk in enumerate(chunks):
        obj = WeaviateObject(properties=_chunk_to_props(chunk, doc_date_iso))
        if i in emb_map:
            nv, emb = emb_map[i]
            if nv == "text_small":
                obj.vector_text_small = emb
            elif nv == "text_large":
                obj.vector_text_large = emb
            elif nv == "image":
                obj.vector_image = emb
        objects.append(obj)

    return objects


# ── Intent-driven parse dispatchers ──────────────────────────────────────────

async def _process_pdf(
    intent: DocumentIntent,
    orig_path: Path,
    images_dir: Path,
    filename: str,
    openrouter_client,
) -> tuple[List[Chunk], Optional[datetime]]:
    parsed = await parse_pdf(orig_path, images_dir)
    chunks: List[Chunk] = []

    chunks.extend(
        route_and_chunk(
            doc_type=intent.doc_type,
            source=intent.source,
            filename=filename,
            original_file_path=str(orig_path),
            text_pages=parsed.text_pages,
            tables=parsed.tables,
        )
    )

    for img in parsed.images:
        img_path = img["file_path"]
        page_no  = img["page_no"]
        idx      = len(chunks)
        caption  = await generate_caption(img_path, openrouter_client)
        chunks.append(chunk_image_caption(
            caption, img_path, page_no=page_no, source=intent.source, chunk_index=idx,
        ))
        chunks.append(Chunk(
            content=caption or f"Image page {page_no}",
            doc_type=intent.doc_type,
            source=intent.source,
            embed_model="clip",
            named_vector="image",
            chunk_type="image_caption",
            page_no=page_no,
            file_path=img_path,
            chunk_index=idx + 1,
        ))

    # Extract date from the first pages text already peeked by the parser
    full_text = " ".join(p.get("content", "") for p in parsed.text_pages[:3])
    doc_date = extract_date_from_pdf_text(full_text)

    logger.info(
        "[PIPELINE] PDF processed | is_legal=%s table_heavy=%s pages=%d chunks=%d doc_date=%s",
        intent.is_legal, intent.is_table_heavy, parsed.page_count, len(chunks), doc_date,
    )
    return chunks, doc_date


async def _process_html(
    intent: DocumentIntent,
    file_bytes: bytes,
    orig_path: Path,
    filename: str,
) -> tuple[List[Chunk], Optional[datetime]]:
    raw_text = parse_html(file_bytes)
    chunks = route_and_chunk(
        doc_type="html",
        source="html",
        filename=filename,
        original_file_path=str(orig_path),
        html_text=raw_text,
    )
    doc_date = extract_date_from_html(file_bytes)
    logger.info("[PIPELINE] HTML processed | chunks=%d doc_date=%s", len(chunks), doc_date)
    return chunks, doc_date


async def _process_image(
    intent: DocumentIntent,
    file_bytes: bytes,
    images_dir: Path,
    filename: str,
    openrouter_client,
) -> tuple[List[Chunk], Optional[datetime]]:
    img_path = save_image_locally(file_bytes, images_dir)
    caption  = await generate_caption(img_path, openrouter_client)
    chunks = [
        chunk_image_caption(caption, img_path, page_no=0, source="image", chunk_index=0),
        Chunk(
            content=caption or f"Image: {filename}",
            doc_type="image", source="image",
            embed_model="clip", named_vector="image",
            chunk_type="image_caption", page_no=0,
            file_path=img_path, chunk_index=1,
        ),
    ]
    logger.info("[PIPELINE] Image processed | caption_len=%d", len(caption))
    return chunks, None   # images don't have an embedded date


async def _process_other(
    intent: DocumentIntent,
    file_bytes: bytes,
    orig_path: Path,
    filename: str,
) -> tuple[List[Chunk], Optional[datetime]]:
    plain_text = file_bytes.decode("utf-8", errors="replace")
    chunks = route_and_chunk(
        doc_type=intent.doc_type,
        source=intent.source,
        filename=filename,
        original_file_path=str(orig_path),
        plain_text=plain_text,
    )
    doc_date = extract_date_from_text(plain_text)
    logger.info("[PIPELINE] Plain-text processed | chunks=%d doc_date=%s", len(chunks), doc_date)
    return chunks, doc_date


def _save_local_metadata(
    user_id: str,
    doc_id: str,
    filename: str,
    doc_type: str,
    doc_date: Optional[datetime],
    doc_date_source: str,
    ingestion_date: datetime,
    chunk_count: int,
    file_path: str,
    intent: DocumentIntent,
) -> str:
    """Write a JSON sidecar at storage/metadata/{user_id}/{doc_id}.json."""
    meta_dir = Path(settings.METADATA_DIR) / user_id
    meta_dir.mkdir(parents=True, exist_ok=True)
    meta_path = meta_dir / f"{doc_id}.json"

    payload = {
        "doc_id":           doc_id,
        "user_id":          user_id,
        "filename":         filename,
        "doc_type":         doc_type,
        "doc_date":         doc_date.isoformat() if doc_date else None,
        "doc_date_source":  doc_date_source,
        "ingestion_date":   ingestion_date.isoformat(),
        "chunk_count":      chunk_count,
        "file_path":        file_path,
        "is_legal":         intent.is_legal,
        "is_table_heavy":   intent.is_table_heavy,
        "is_scanned":       intent.is_scanned,
        "language":         intent.language,
        "embed_model":      intent.embed_model,
        "named_vector":     intent.named_vector,
        "chunking_strategy":intent.chunking_strategy,
    }

    meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("[PIPELINE] Metadata saved | path=%s", meta_path)
    return str(meta_path)


# ── Public entry point ────────────────────────────────────────────────────────

async def run_ingestion_pipeline(
    *,
    file_bytes: bytes,
    filename: str,
    user_id: str,
    doc_id: str,
    doc_type_hint: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Intent-driven ingestion pipeline.

    Returns:
        {
          "chunk_count":      int,
          "status":           "done" | "error",
          "error":            str | None,
          "doc_date":         datetime | None,
          "doc_date_source":  "extracted" | "ingestion",
          "metadata_path":    str,
          "metadata_json":    str (JSON),
          "intent":           dict
        }
    """
    ingestion_date = datetime.now(timezone.utc)

    # ── Step 1: Intent detection ───────────────────────────────────────────────
    intent = await detect_intent(filename, file_bytes, doc_type_hint=doc_type_hint)
    logger.info(
        "[PIPELINE] Intent detected | doc_type=%s chunking=%s embed=%s "
        "legal=%s table_heavy=%s scanned=%s confidence=%.2f",
        intent.doc_type, intent.chunking_strategy, intent.embed_model,
        intent.is_legal, intent.is_table_heavy, intent.is_scanned, intent.confidence,
    )

    # ── Step 2: Persist file locally ──────────────────────────────────────────
    storage_root  = Path(settings.STORAGE_DIR) / user_id / doc_id
    originals_dir = storage_root / "originals"
    images_dir    = storage_root / "images"
    originals_dir.mkdir(parents=True, exist_ok=True)

    orig_path = originals_dir / filename
    orig_path.write_bytes(file_bytes)

    # ── Step 3: Ensure Weaviate tenant exists ──────────────────────────────────
    await asyncio.to_thread(ensure_tenant, user_id)

    openrouter_client = await _get_openrouter_client()
    doc_date: Optional[datetime] = None
    doc_date_source = "ingestion"

    try:
        # ── Step 4: Parse & chunk based on intent ──────────────────────────────
        doc_type = intent.doc_type

        if doc_type in ("pdf", "legal", "news", "medical", "financial", "research"):
            all_chunks, doc_date = await _process_pdf(
                intent, orig_path, images_dir, filename, openrouter_client
            )
        elif doc_type == "html":
            all_chunks, doc_date = await _process_html(intent, file_bytes, orig_path, filename)
        elif doc_type == "image":
            all_chunks, doc_date = await _process_image(
                intent, file_bytes, images_dir, filename, openrouter_client
            )
        else:
            all_chunks, doc_date = await _process_other(intent, file_bytes, orig_path, filename)

        if doc_date:
            doc_date_source = "extracted"

        effective_date = doc_date or ingestion_date
        doc_date_iso   = effective_date.isoformat()

        if not all_chunks:
            logger.warning("[PIPELINE] No chunks produced | doc_id=%s", doc_id)

        # ── Step 5: Embed (batched, parallel models) ───────────────────────────
        weaviate_objects = await _embed_and_build_objects(all_chunks, doc_date_iso)

        for obj, chunk in zip(weaviate_objects, all_chunks):
            obj.properties["doc_id"]   = doc_id
            obj.properties["filename"] = filename
            obj.properties["user_id"]  = user_id

        # ── Step 6: Batch upsert to Weaviate ──────────────────────────────────
        result = await batch_upsert(weaviate_objects, user_id)

        chunk_count = result["upserted"]

        # ── Step 7: Save local metadata JSON ──────────────────────────────────
        meta_path = _save_local_metadata(
            user_id=user_id, doc_id=doc_id, filename=filename,
            doc_type=doc_type, doc_date=doc_date,
            doc_date_source=doc_date_source,
            ingestion_date=ingestion_date,
            chunk_count=chunk_count,
            file_path=str(orig_path),
            intent=intent,
        )

        metadata_payload = {
            "doc_date_source": doc_date_source,
            "language": intent.language,
            "is_legal": intent.is_legal,
            "is_table_heavy": intent.is_table_heavy,
        }

        logger.info(
            "[PIPELINE] Complete | user=%s doc_id=%s chunks=%d upserted=%d errors=%d doc_date=%s",
            user_id, doc_id, len(all_chunks), result["upserted"], result["errors"], doc_date,
        )
        return {
            "chunk_count":     chunk_count,
            "status":          "done",
            "error":           None,
            "doc_date":        doc_date,
            "doc_date_source": doc_date_source,
            "metadata_path":   meta_path,
            "metadata_json":   json.dumps(metadata_payload),
            "intent": {
                "doc_type":          intent.doc_type,
                "chunking_strategy": intent.chunking_strategy,
                "embed_model":       intent.embed_model,
                "named_vector":      intent.named_vector,
                "is_legal":          intent.is_legal,
                "is_table_heavy":    intent.is_table_heavy,
                "is_scanned":        intent.is_scanned,
                "confidence":        round(intent.confidence, 3),
                "detection_method":  intent.detection_method,
                "reason":            intent.reason,
            },
        }

    except Exception as e:
        logger.exception("[PIPELINE] Failed | user=%s doc_id=%s err=%s", user_id, doc_id, e)
        return {
            "chunk_count":     0,
            "status":          "error",
            "error":           str(e),
            "doc_date":        None,
            "doc_date_source": "ingestion",
            "metadata_path":   "",
            "metadata_json":   "{}",
            "intent":          {"doc_type": intent.doc_type, "detection_method": intent.detection_method},
        }
