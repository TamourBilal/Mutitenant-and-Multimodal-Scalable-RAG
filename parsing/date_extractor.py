"""
Extract a publication / creation date from document content.

Tries document-specific signals first (HTML meta tags, PDF text),
then falls back to regex over the first few thousand characters of text.
Returns None if no date can be reliably extracted; callers should fall
back to the ingestion timestamp in that case.
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Optional

# strptime format strings tried in order; stop at first match
_FORMATS = [
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%d-%m-%Y",
    "%d/%m/%Y",
    "%m-%d-%Y",
    "%m/%d/%Y",
    "%B %d, %Y",
    "%B %d %Y",
    "%b %d, %Y",
    "%b %d %Y",
    "%d %B %Y",
    "%d %b %Y",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%dT%H:%M:%S%z",
]

# Regex patterns that capture a date-like string from free text
_TEXT_PATTERNS = [
    r'\b(\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2}:\d{2}(?:Z|[+-]\d{2}:\d{2})?)?)\b',
    r'\b(\d{2}/\d{2}/\d{4})\b',
    r'\b(\d{2}-\d{2}-\d{4})\b',
    (
        r'\b((?:January|February|March|April|May|June|July|August|'
        r'September|October|November|December)\s+\d{1,2},?\s+\d{4})\b'
    ),
    (
        r'\b(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|'
        r'September|October|November|December)\s+\d{4})\b'
    ),
    (
        r'\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?\s+\d{1,2},?\s+\d{4})\b'
    ),
]

# HTML meta / time attributes that carry a publication date
_HTML_META_RE = re.compile(
    r'(?:'
    r'(?:property|name)=["\'](?:article:published_time|datePublished|pubdate|'
    r'DC\.date|og:article:published_time|article:modified_time)["\'][^>]+'
    r'content=["\']([^"\']{4,40})["\']'
    r'|'
    r'content=["\']([^"\']{4,40})["\'][^>]+'
    r'(?:property|name)=["\'](?:article:published_time|datePublished|pubdate|DC\.date)["\']'
    r'|'
    r'<time[^>]+datetime=["\']([^"\']{4,40})["\']'
    r')',
    re.IGNORECASE,
)


def _parse_candidate(s: str) -> Optional[datetime]:
    """Try each strptime format; reject dates outside 1990–2030."""
    s = s.strip().rstrip(",").replace("Z", "").split("+")[0].split("-")[0] if "T" not in s else s
    # Restore ISO separator if it was trimmed
    raw = s.strip().rstrip(",")
    for fmt in _FORMATS:
        try:
            dt = datetime.strptime(raw, fmt)
            if 1990 <= dt.year <= 2030:
                return dt
        except ValueError:
            continue
    # For ISO strings with timezone offset, strip and retry
    iso_clean = re.sub(r'([+-]\d{2}:\d{2}|Z)$', '', raw)
    if iso_clean != raw:
        for fmt in _FORMATS:
            try:
                dt = datetime.strptime(iso_clean, fmt)
                if 1990 <= dt.year <= 2030:
                    return dt
            except ValueError:
                continue
    return None


def extract_date_from_text(text: str, scan_chars: int = 4000) -> Optional[datetime]:
    """Scan the first `scan_chars` characters of plain text for a date."""
    sample = text[:scan_chars]
    for pattern in _TEXT_PATTERNS:
        m = re.search(pattern, sample, re.IGNORECASE)
        if m:
            candidate = next(g for g in m.groups() if g) if m.groups() else m.group(0)
            dt = _parse_candidate(candidate)
            if dt:
                return dt
    return None


def extract_date_from_html(html_bytes: bytes) -> Optional[datetime]:
    """Try HTML meta tags first, then fall through to text scan."""
    try:
        html = html_bytes[:10_000].decode("utf-8", errors="replace")
    except Exception:
        return None

    m = _HTML_META_RE.search(html)
    if m:
        candidate = next((g for g in m.groups() if g), None)
        if candidate:
            dt = _parse_candidate(candidate)
            if dt:
                return dt

    return extract_date_from_text(html)


def extract_date_from_pdf_text(text_sample: str) -> Optional[datetime]:
    """Extract date from the text peek already obtained by the PDF parser."""
    return extract_date_from_text(text_sample, scan_chars=5000)
