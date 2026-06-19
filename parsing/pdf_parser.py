"""
PDF parser — PyMuPDF (fitz) for text, tables, and images.
Fast and reliable — no pdfplumber dependency.
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ParsedPDF:
    text_pages: List[Dict[str, Any]] = field(default_factory=list)
    tables: List[Dict[str, Any]] = field(default_factory=list)
    images: List[Dict[str, Any]] = field(default_factory=list)
    page_count: int = 0


def _table_to_markdown(table: List[List[str]]) -> str:
    if not table or not table[0]:
        return ""
    header = "| " + " | ".join(str(c or "") for c in table[0]) + " |"
    sep    = "| " + " | ".join("---" for _ in table[0]) + " |"
    rows   = ["| " + " | ".join(str(c or "") for c in row) + " |" for row in table[1:]]
    return "\n".join([header, sep] + rows)


def _extract_all(file_path: Path, image_dir: Path) -> Dict[str, Any]:
    """
    Extract text, tables (heuristic), and images from a PDF using PyMuPDF.
    Runs in a thread — no async inside.
    """
    import fitz  # PyMuPDF

    text_pages: List[Dict] = []
    tables: List[Dict] = []
    images: List[Dict] = []

    image_dir.mkdir(parents=True, exist_ok=True)

    with fitz.open(str(file_path)) as doc:
        page_count = len(doc)
        max_pages = 20   # DEMO LIMIT — hard cap at 20 pages
        logger.info("[PDF_PARSER] Opened | total_pages=%d processing=%d file=%s", page_count, max_pages, file_path.name)

        for page_idx in range(max_pages):
            page = doc[page_idx]
            page_no = page_idx + 1

            # ── Text ──────────────────────────────────────────────────────────
            try:
                text = page.get_text("text").strip()
                if text:
                    text_pages.append({"page_no": page_no, "content": text})
                logger.info(
                    "[PDF_PARSER] Page %d/%d | chars=%d tables_so_far=%d images_so_far=%d",
                    page_no, max_pages, len(text), len(tables), len(images),
                )
            except Exception as e:
                logger.warning("[PDF_PARSER] text failed page=%d err=%s", page_no, e)

            # ── Tables (heuristic: detect grid-like text blocks) ───────────────
            try:
                blocks = page.get_text("blocks")
                # Simple heuristic: lines with 3+ tab/space separated columns
                table_lines = []
                for block in blocks:
                    block_text = block[4] if len(block) > 4 else ""
                    for line in block_text.splitlines():
                        parts = [p.strip() for p in line.split("  ") if p.strip()]
                        if len(parts) >= 3:
                            table_lines.append(parts)
                if len(table_lines) >= 2:
                    md = _table_to_markdown(table_lines)
                    if md:
                        tables.append({"page_no": page_no, "content": md})
            except Exception as e:
                logger.debug("[PDF_PARSER] table heuristic failed page=%d err=%s", page_no, e)

            # ── Images ────────────────────────────────────────────────────────
            try:
                for img_info in page.get_images(full=False):
                    base = doc.extract_image(img_info[0]) or {}
                    img_bytes = base.get("image", b"")
                    if not img_bytes or len(img_bytes) < 2048:
                        continue
                    ext   = str(base.get("ext") or "png")
                    fname = f"{uuid.uuid4().hex}.{ext}"
                    dest  = image_dir / fname
                    dest.write_bytes(img_bytes)
                    images.append({"page_no": page_no, "file_path": str(dest)})
            except Exception as e:
                logger.debug("[PDF_PARSER] image skip page=%d err=%s", page_no, e)

    logger.info(
        "[PDF_PARSER] Done | total_pages=%d processed=%d text=%d tables=%d images=%d",
        page_count, max_pages, len(text_pages), len(tables), len(images),
    )
    return {
        "text_pages": text_pages,
        "tables":     tables,
        "images":     images,
        "page_count": max_pages,
    }


async def parse_pdf(file_path: Path, image_dir: Path) -> ParsedPDF:
    """Async wrapper — runs PyMuPDF extraction in a thread pool."""
    try:
        logger.info("[PDF_PARSER] Starting extraction | file=%s", file_path.name)
        result = await asyncio.wait_for(
            asyncio.to_thread(_extract_all, file_path, image_dir),
            timeout=120,
        )
        return ParsedPDF(
            text_pages=result["text_pages"],
            tables=result["tables"],
            images=result["images"],
            page_count=result["page_count"],
        )
    except asyncio.TimeoutError:
        logger.error("[PDF_PARSER] Timed out after 120s | file=%s", file_path.name)
        return ParsedPDF()
    except Exception as e:
        logger.error("[PDF_PARSER] Failed | file=%s err=%s", file_path.name, e)
        return ParsedPDF()
