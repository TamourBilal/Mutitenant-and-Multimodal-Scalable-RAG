"""
PDF parser — PyMuPDF (fitz) for text, tables, and images.
Fast and reliable — no pdfplumber dependency.
"""
from __future__ import annotations

import asyncio
import logging
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
# A "value" token: number (with spaces as thousands sep handled by row join),
# percentage, ratio like "47.7 / 62.7", ellipsis, or footnote letter.
_VALUE_RE = re.compile(r"^[\d.,/%()\-–\s]+$")


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


def _extract_tables(page, page_no: int) -> tuple[List[Dict], List[Any]]:
    """
    Detect *ruled* tables with PyMuPDF's built-in finder and render each as
    clean markdown. Returns (table_dicts, table_bboxes).

    We deliberately use only the `lines_strict` strategy. The `text` /
    `lines` strategies over-detect on borderless layouts — they grab the whole
    page as one "table" and split words mid-token, which corrupts both the
    table markdown and (when the region is excluded) the narrative text. For
    such documents the plain page text is the better representation, so when no
    ruled table exists we simply return nothing and keep the full text.
    """
    tables: List[Dict] = []
    bboxes: List[Any] = []

    finder = getattr(page, "find_tables", None)
    if finder is None:
        return tables, bboxes

    try:
        found = page.find_tables(strategy="lines_strict")
        for tab in found.tables:
            # Require a genuine grid: at least 2 columns and 2 rows.
            if tab.col_count < 2 or tab.row_count < 2:
                continue
            try:
                md = tab.to_markdown(clean=True).strip()
            except Exception:
                md = _table_to_markdown(tab.extract())
            if md and md.count("|") >= 4:
                tables.append({"page_no": page_no, "content": md})
                bboxes.append(tab.bbox)
    except Exception as e:
        logger.debug("[PDF_PARSER] find_tables failed page=%d err=%s", page_no, e)

    return tables, bboxes


def _page_rows(page) -> List[str]:
    """
    Reconstruct visual rows from word coordinates. `get_text("text")` emits one
    token per line for borderless tables, which disconnects each label from its
    values. Grouping words by their y-coordinate rebuilds readable rows like
    'GDP per capita (current US$) 7 579.8 5 928.9 6 707.0'.
    """
    words = page.get_text("words")  # (x0, y0, x1, y1, word, block, line, wordno)
    if not words:
        return [l for l in page.get_text("text").splitlines() if l.strip()]

    rows: Dict[int, List[tuple]] = {}
    for w in words:
        x0, y0, word = w[0], w[1], w[4]
        key = next((k for k in rows if abs(k - y0) <= 3), round(y0))
        rows.setdefault(key, []).append((x0, word))

    lines: List[str] = []
    for y in sorted(rows):
        line = " ".join(word for _, word in sorted(rows[y], key=lambda t: t[0])).strip()
        if line:
            lines.append(line)
    return lines


def _reconstruct_stat_table(lines: List[str], page_no: int) -> Optional[Dict]:
    """
    Build a markdown table from row-grouped statistical lines. A page qualifies
    if it has a header row containing >=2 year columns (e.g. '2010 2015 2023').
    Each data row becomes 'Indicator | v1 | v2 | v3'. Returns None if the page
    is not a statistical table.
    """
    year_headers = [l for l in lines if len(_YEAR_RE.findall(l)) >= 2]
    if not year_headers:
        return None

    years = _YEAR_RE.findall(" ".join(year_headers))  # list of '20'/'19' prefixes — count only
    n_cols = len(_YEAR_RE.findall(year_headers[0]))
    if n_cols < 2:
        return None

    md_rows: List[str] = []
    for line in lines:
        nums = len(re.findall(r"\d", line))
        if nums == 0:
            continue
        # Split label from the trailing numeric value group. Allow an optional
        # single footnote letter between label and values (e.g. '... %) d 2.8 ...')
        # and trailing footnote letters/commas after the values (e.g. 'a,b').
        m = re.search(
            r"^(.*?)(?:\s[a-z])?\s((?:[\d][\d .,/%()=–-]*\s*)+(?:[a-z](?:,[a-z])*)?)$",
            line,
        )
        if m and any(ch.isdigit() for ch in m.group(2)):
            label = m.group(1).strip()
            values = m.group(2).strip()
            md_rows.append(f"| {label} | {values} |")
        else:
            md_rows.append(f"| {line.strip()} |  |")

    if len(md_rows) < 3:
        return None

    header = "| Indicator | Values (by year columns) |\n| --- | --- |"
    md = header + "\n" + "\n".join(md_rows)
    return {"page_no": page_no, "content": md}


def _extract_all(file_path: Path, image_dir: Path) -> Dict[str, Any]:
    """
    Extract text, tables, and images from a PDF using PyMuPDF.
    Tables use the native table finder; narrative text excludes table regions.
    Runs in a thread — no async inside.
    """
    import fitz  # PyMuPDF
    from config import settings

    text_pages: List[Dict] = []
    tables: List[Dict] = []
    images: List[Dict] = []

    image_dir.mkdir(parents=True, exist_ok=True)

    with fitz.open(str(file_path)) as doc:
        page_count = len(doc)
        cap = settings.PDF_MAX_PAGES or page_count
        max_pages = min(page_count, cap)
        logger.info("[PDF_PARSER] Opened | total_pages=%d processing=%d file=%s", page_count, max_pages, file_path.name)

        for page_idx in range(max_pages):
            page = doc[page_idx]
            page_no = page_idx + 1

            # ── Row-grouped text (keeps label + values together) ──────────────
            try:
                rows = _page_rows(page)
                text = "\n".join(rows).strip()
                if text:
                    text_pages.append({"page_no": page_no, "content": text})
            except Exception as e:
                logger.warning("[PDF_PARSER] text failed page=%d err=%s", page_no, e)
                rows, text = [], ""

            # ── Tables: native ruled tables + reconstructed statistical tables ─
            page_tables, _ = _extract_tables(page, page_no)
            stat_table = _reconstruct_stat_table(rows, page_no) if rows else None
            if stat_table:
                page_tables.append(stat_table)
            tables.extend(page_tables)

            logger.info(
                "[PDF_PARSER] Page %d/%d | chars=%d rows=%d tables_on_page=%d images_so_far=%d",
                page_no, max_pages, len(text), len(rows), len(page_tables), len(images),
            )

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
