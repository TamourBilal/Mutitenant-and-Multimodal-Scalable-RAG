"""
PDF parser — pdfplumber (text + tables) + PyMuPDF (raster images).
Saves images locally. No S3 dependency.
Adapted from the original document_parser.py.
"""
from __future__ import annotations

import asyncio
import logging
import math
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ParsedPDF:
    text_pages: List[Dict[str, Any]] = field(default_factory=list)
    # [{"page_no": int, "content": str}]
    tables: List[Dict[str, Any]] = field(default_factory=list)
    # [{"page_no": int, "content": str}]  — markdown rows
    images: List[Dict[str, Any]] = field(default_factory=list)
    # [{"page_no": int, "file_path": str}]
    page_count: int = 0


def _table_to_markdown(table: List[List[Optional[str]]]) -> str:
    if not table or not table[0]:
        return ""
    header = "| " + " | ".join(str(c or "") for c in table[0]) + " |"
    sep = "| " + " | ".join("---" for _ in table[0]) + " |"
    rows = [
        "| " + " | ".join(str(c or "") for c in row) + " |"
        for row in table[1:]
    ]
    return "\n".join([header, sep] + rows)


def _extract_text_and_tables(
    file_path: Path,
    start: int,
    end: int,
) -> List[Dict[str, Any]]:
    """Returns list of {page_no, text_content, table_content} per page."""
    import pdfplumber

    results = []
    with pdfplumber.open(str(file_path)) as pdf:
        for idx in range(start, end):
            page = pdf.pages[idx]
            page_no = idx + 1

            text = ""
            try:
                raw = page.extract_text() or ""
                text = raw.strip()
            except Exception:
                pass

            tables: List[str] = []
            try:
                for t in page.extract_tables() or []:
                    md = _table_to_markdown(t)
                    if md:
                        tables.append(md)
            except Exception:
                pass

            results.append(
                {
                    "page_no": page_no,
                    "text_content": text,
                    "table_content": tables,
                }
            )
    return results


def _extract_images(
    file_path: Path,
    image_dir: Path,
    start: int,
    end: int,
) -> List[Dict[str, Any]]:
    """Extract raster images using PyMuPDF, save to local dir."""
    import fitz  # PyMuPDF

    image_dir.mkdir(parents=True, exist_ok=True)
    results = []

    with fitz.open(str(file_path)) as doc:
        for page_idx in range(start, end):
            page = doc[page_idx]
            page_no = page_idx + 1

            for img_info in page.get_images(full=False) or []:
                try:
                    base = doc.extract_image(img_info[0]) or {}
                    img_bytes = base.get("image", b"")
                    if not img_bytes or len(img_bytes) < 2048:
                        continue
                    ext = str(base.get("ext") or "png")
                    fname = f"{uuid.uuid4().hex}.{ext}"
                    dest = image_dir / fname
                    dest.write_bytes(img_bytes)
                    results.append({"page_no": page_no, "file_path": str(dest)})
                except Exception as e:
                    logger.debug("[PDF_PARSER] image skip page=%d err=%s", page_no, e)
    return results


async def parse_pdf(
    file_path: Path,
    image_dir: Path,
    *,
    max_workers: int = 3,
) -> ParsedPDF:
    """
    Async PDF parse. Returns text pages, table pages, and image file paths.
    Text and tables are separate so different chunking strategies can be applied.
    """
    try:
        import pdfplumber
        with pdfplumber.open(str(file_path)) as pdf:
            page_count = len(pdf.pages)
    except Exception as e:
        logger.error("[PDF_PARSER] Failed to open | path=%s err=%s", file_path, e)
        return ParsedPDF()

    if page_count == 0:
        return ParsedPDF()

    workers = min(max_workers, page_count)
    chunk_size = math.ceil(page_count / workers)
    ranges = [(s, min(page_count, s + chunk_size)) for s in range(0, page_count, chunk_size)]

    # Parallel text/table extraction
    text_tasks = [
        asyncio.to_thread(_extract_text_and_tables, file_path, s, e)
        for s, e in ranges
    ]
    # Parallel image extraction
    img_tasks = [
        asyncio.to_thread(_extract_images, file_path, image_dir, s, e)
        for s, e in ranges
    ]

    all_results = await asyncio.gather(*text_tasks, *img_tasks, return_exceptions=True)
    text_results = all_results[:len(ranges)]
    img_results = all_results[len(ranges):]

    text_pages: List[Dict[str, Any]] = []
    tables: List[Dict[str, Any]] = []
    images: List[Dict[str, Any]] = []

    for chunk in text_results:
        if isinstance(chunk, Exception):
            logger.warning("[PDF_PARSER] text chunk error: %s", chunk)
            continue
        for page in chunk:
            if page["text_content"]:
                text_pages.append({"page_no": page["page_no"], "content": page["text_content"]})
            for tbl in page["table_content"]:
                tables.append({"page_no": page["page_no"], "content": tbl})

    for chunk in img_results:
        if isinstance(chunk, Exception):
            logger.warning("[PDF_PARSER] image chunk error: %s", chunk)
            continue
        images.extend(chunk)

    text_pages.sort(key=lambda x: x["page_no"])
    tables.sort(key=lambda x: x["page_no"])

    logger.info(
        "[PDF_PARSER] Done | pages=%d text_pages=%d tables=%d images=%d",
        page_count, len(text_pages), len(tables), len(images),
    )
    return ParsedPDF(
        text_pages=text_pages,
        tables=tables,
        images=images,
        page_count=page_count,
    )
