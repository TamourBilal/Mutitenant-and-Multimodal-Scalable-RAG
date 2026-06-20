from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Literal, Optional

from config import settings

logger = logging.getLogger(__name__)

EmbedModel = Literal["small", "large", "clip"]
ChunkType = Literal["text", "table", "image_caption"]
NamedVector = Literal["text_small", "text_large", "image"]


@dataclass
class Chunk:
    content: str
    doc_type: str           # legal | pdf | html | image | other
    source: str             # pdf | html | image | other
    embed_model: EmbedModel # small | large | clip
    named_vector: NamedVector
    chunk_type: ChunkType   # text | table | image_caption
    page_no: int            # 0 for non-PDF content
    file_path: str          # local path (images: path to image file; text: path to original doc)
    chunk_index: int        # position within this document


def _is_legal(doc_type: str, filename: str) -> bool:
    if doc_type == "legal":
        return True
    name_lower = filename.lower()
    return any(kw in name_lower for kw in settings.LEGAL_KEYWORDS)


def chunk_text_pages(
    text_pages: List[dict],
    *,
    doc_type: str,
    source: str,
    filename: str,
    original_file_path: str,
) -> List[Chunk]:
    """Chunk text pages from a parsed PDF."""
    from chunking.semantic_chunker import split_semantic

    legal = _is_legal(doc_type, filename)
    model: EmbedModel = "large" if legal else "small"
    nvec: NamedVector = "text_large" if legal else "text_small"
    effective_doc_type = "legal" if legal else doc_type

    chunks: List[Chunk] = []
    idx = 0
    for page in text_pages:
        page_no = page.get("page_no", 0)
        content = page.get("content", "").strip()
        if not content:
            continue
        for piece in split_semantic(content, model=model):
            chunks.append(
                Chunk(
                    content=piece,
                    doc_type=effective_doc_type,
                    source=source,
                    embed_model=model,
                    named_vector=nvec,
                    chunk_type="text",
                    page_no=page_no,
                    file_path=original_file_path,
                    chunk_index=idx,
                )
            )
            idx += 1
    return chunks


def chunk_tables(
    tables: List[dict],
    *,
    doc_type: str,
    source: str,
    filename: str,
    original_file_path: str,
) -> List[Chunk]:
    """Chunk table markdown — always uses text_small (exact values, BM25 benefits)."""
    from chunking.semantic_chunker import split_recursive

    legal = _is_legal(doc_type, filename)
    model: EmbedModel = "large" if legal else "small"
    nvec: NamedVector = "text_large" if legal else "text_small"
    effective_doc_type = "legal" if legal else doc_type

    chunks: List[Chunk] = []
    idx = 0
    for table in tables:
        page_no = table.get("page_no", 0)
        content = table.get("content", "").strip()
        if not content:
            continue
        for piece in split_recursive(content):
            chunks.append(
                Chunk(
                    content=piece,
                    doc_type=effective_doc_type,
                    source=source,
                    embed_model=model,
                    named_vector=nvec,
                    chunk_type="table",
                    page_no=page_no,
                    file_path=original_file_path,
                    chunk_index=idx,
                )
            )
            idx += 1
    return chunks


def chunk_html(
    text: str,
    *,
    source: str,
    original_file_path: str,
) -> List[Chunk]:
    from chunking.recursive_html import split_html

    chunks: List[Chunk] = []
    for idx, piece in enumerate(split_html(text)):
        chunks.append(
            Chunk(
                content=piece,
                doc_type="html",
                source=source,
                embed_model="small",
                named_vector="text_small",
                chunk_type="text",
                page_no=0,
                file_path=original_file_path,
                chunk_index=idx,
            )
        )
    return chunks


def chunk_image_caption(
    caption: str,
    image_file_path: str,
    *,
    page_no: int = 0,
    source: str = "image",
    chunk_index: int = 0,
) -> Chunk:
    """
    Single chunk for an image.
    Caption is embedded with text-embedding-3-small → text_small (1536-dim).
    No local CLIP model needed.
    """
    return Chunk(
        content=caption if caption else f"Image: {image_file_path}",
        doc_type="image",
        source=source,
        embed_model="small",
        named_vector="text_small",
        chunk_type="image_caption",
        page_no=page_no,
        file_path=image_file_path,
        chunk_index=chunk_index,
    )


def route_and_chunk(
    *,
    doc_type: str,
    source: str,
    filename: str,
    original_file_path: str,
    # For PDF
    text_pages: Optional[List[dict]] = None,
    tables: Optional[List[dict]] = None,
    # For HTML
    html_text: Optional[str] = None,
    # For plain text / other
    plain_text: Optional[str] = None,
) -> List[Chunk]:
    """
    Route content to the correct chunker based on doc_type.
    Returns a flat list of Chunk objects ready for embedding.
    """
    chunks: List[Chunk] = []

    if doc_type in ("pdf", "docx", "legal", "news", "medical", "financial", "research", "other") or source == "pdf":
        logger.info("[ROUTER] PDF branch | doc_type=%s text_pages=%d tables=%d", doc_type, len(text_pages or []), len(tables or []))
        if text_pages:
            chunks.extend(
                chunk_text_pages(
                    text_pages,
                    doc_type=doc_type,
                    source=source,
                    filename=filename,
                    original_file_path=original_file_path,
                )
            )
        if tables:
            t_chunks = chunk_tables(
                tables,
                doc_type=doc_type,
                source=source,
                filename=filename,
                original_file_path=original_file_path,
            )
            # Re-index after text chunks
            offset = len(chunks)
            for c in t_chunks:
                c.chunk_index += offset
            chunks.extend(t_chunks)

    elif doc_type == "html":
        text = html_text or plain_text or ""
        if text:
            chunks.extend(
                chunk_html(
                    text,
                    source=source,
                    original_file_path=original_file_path,
                )
            )

    elif plain_text:
        # Generic fallback — recursive split
        from chunking.recursive_html import split_html
        for idx, piece in enumerate(split_html(plain_text)):
            chunks.append(
                Chunk(
                    content=piece,
                    doc_type=doc_type,
                    source=source,
                    embed_model="small",
                    named_vector="text_small",
                    chunk_type="text",
                    page_no=0,
                    file_path=original_file_path,
                    chunk_index=idx,
                )
            )

    logger.info(
        "[ROUTER] Chunks created | doc_type=%s filename=%s count=%d",
        doc_type, filename, len(chunks),
    )
    return chunks
