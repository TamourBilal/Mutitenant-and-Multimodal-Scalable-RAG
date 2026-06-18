"""
Intent detection layer.

Determines how a document should be processed BEFORE parsing begins.

Detection is a two-pass system:
  Pass 1 — Fast heuristics (extension + filename keywords + magic bytes + content peek).
            If confidence ≥ 0.85 we stop here. No LLM call.
  Pass 2 — LLM classification (OpenRouter gpt-4o-mini) using a content sample.
            Used only when heuristics are uncertain.

The returned DocumentIntent drives every downstream decision:
  chunking strategy, embedding model, named vector, special processing flags.
"""
from __future__ import annotations

import io
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional

from config import settings

logger = logging.getLogger(__name__)

# ── Types ──────────────────────────────────────────────────────────────────────

DocType = Literal["legal", "news", "medical", "financial", "research", "pdf", "html", "image", "other"]
ChunkingStrategy = Literal["semantic", "recursive", "image_caption", "none"]
EmbedModel = Literal["small", "large", "clip"]
NamedVector = Literal["text_small", "text_large", "image"]


@dataclass
class DocumentIntent:
    """Full processing specification derived from intent detection."""

    # Core classification
    doc_type: DocType = "other"
    source: str = "other"            # "pdf" | "html" | "image" | "other"

    # Chunking
    chunking_strategy: ChunkingStrategy = "recursive"

    # Embedding
    embed_model: EmbedModel = "small"
    named_vector: NamedVector = "text_small"

    # Processing hints
    is_legal: bool = False
    is_table_heavy: bool = False     # >30 % of pages have tables → dedicate semantic pass to tables
    is_scanned: bool = False         # PDF with no text layer → future OCR hook
    is_mixed_content: bool = False   # PDF with both text and significant images
    language: str = "en"

    # Diagnostics
    confidence: float = 1.0          # 0–1; below 0.85 triggers LLM pass
    detection_method: str = "heuristic"
    reason: str = ""

    # Raw content sample used for detection (not stored in Weaviate)
    content_sample: str = ""


# ── Extension maps ─────────────────────────────────────────────────────────────

_PDF_EXTENSIONS   = {".pdf", ".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls", ".odt"}
_HTML_EXTENSIONS  = {".html", ".htm", ".xhtml", ".xml"}
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".tif", ".svg"}
_TEXT_EXTENSIONS  = {".txt", ".md", ".rst", ".csv", ".tsv", ".json", ".yaml", ".yml"}

_LEGAL_KEYWORDS: set = {kw.lower() for kw in settings.LEGAL_KEYWORDS}
_NEWS_KEYWORDS: set = {kw.lower() for kw in settings.NEWS_KEYWORDS}
_MEDICAL_KEYWORDS: set = {kw.lower() for kw in settings.MEDICAL_KEYWORDS}
_FINANCIAL_KEYWORDS: set = {kw.lower() for kw in settings.FINANCIAL_KEYWORDS}
_RESEARCH_KEYWORDS: set = {kw.lower() for kw in settings.RESEARCH_KEYWORDS}


def _semantic_doc_type(filename: str, text_sample: str) -> Optional[str]:
    """Keyword-based semantic classification for news/medical/financial/research."""
    name_and_text = (filename.lower() + " " + text_sample.lower())
    scores = {
        "news":       sum(1 for kw in _NEWS_KEYWORDS       if kw in name_and_text),
        "medical":    sum(1 for kw in _MEDICAL_KEYWORDS    if kw in name_and_text),
        "financial":  sum(1 for kw in _FINANCIAL_KEYWORDS  if kw in name_and_text),
        "research":   sum(1 for kw in _RESEARCH_KEYWORDS   if kw in name_and_text),
    }
    best, count = max(scores.items(), key=lambda x: x[1])
    return best if count >= 2 else None

# Magic bytes: first 8 bytes → file type
_MAGIC: Dict[bytes, str] = {
    b"%PDF":     "pdf",
    b"\x89PNG":  "image",
    b"\xff\xd8": "image",   # JPEG
    b"GIF8":     "image",
    b"BM":       "image",   # BMP
    b"RIFF":     "image",   # WebP (RIFF....WEBP)
    b"PK\x03\x04": "pdf",   # ZIP-based (docx/xlsx)
    b"<html":    "html",
    b"<!DOC":    "html",
    b"<?xml":    "html",
}


# ── Pass 1: Heuristics ─────────────────────────────────────────────────────────

def _magic_type(file_bytes: bytes) -> Optional[str]:
    head = file_bytes[:8].lower() if len(file_bytes) >= 8 else b""
    for magic, ftype in _MAGIC.items():
        if head.startswith(magic.lower()):
            return ftype
    return None


def _extension_type(filename: str) -> Optional[str]:
    ext = Path(filename).suffix.lower()
    if ext in _PDF_EXTENSIONS:
        return "pdf"
    if ext in _HTML_EXTENSIONS:
        return "html"
    if ext in _IMAGE_EXTENSIONS:
        return "image"
    if ext in _TEXT_EXTENSIONS:
        return "other"
    return None


def _is_legal_name(filename: str) -> bool:
    name = filename.lower()
    return any(kw in name for kw in _LEGAL_KEYWORDS)


def _peek_pdf(file_bytes: bytes) -> Dict:
    """
    Quick peek at PDF content: extract text from first 3 pages.
    Returns {text_sample, table_page_count, total_pages, avg_text_density}.
    """
    try:
        import pdfplumber

        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            total = len(pdf.pages)
            sample_pages = pdf.pages[:min(3, total)]
            texts, table_count = [], 0
            for page in sample_pages:
                t = (page.extract_text() or "").strip()
                texts.append(t)
                if page.extract_tables():
                    table_count += 1
            sample_text = "\n".join(texts)
            avg_density = len(sample_text) / max(1, len(sample_pages))
            return {
                "text_sample": sample_text[:2000],
                "table_page_count": table_count,
                "total_pages": total,
                "avg_text_density": avg_density,
                "sample_page_count": len(sample_pages),
            }
    except Exception as e:
        logger.debug("[INTENT] PDF peek failed: %s", e)
        return {"text_sample": "", "table_page_count": 0, "total_pages": 0, "avg_text_density": 0, "sample_page_count": 0}


def _peek_html(file_bytes: bytes) -> str:
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(file_bytes[:4000].decode("utf-8", errors="replace"), "lxml")
        for tag in soup(["script", "style"]):
            tag.decompose()
        return (soup.get_text(separator=" ", strip=True) or "")[:1500]
    except Exception:
        return file_bytes[:500].decode("utf-8", errors="replace")


def _heuristic_pass(
    filename: str,
    file_bytes: bytes,
) -> DocumentIntent:
    """Return a DocumentIntent from extension + magic bytes + minimal content peek."""

    intent = DocumentIntent()

    # ── Step 1: determine base type ───────────────────────────────────────────
    magic   = _magic_type(file_bytes)
    ext_type = _extension_type(filename)

    raw_type = magic or ext_type or "other"
    intent.source = raw_type  # will be updated below

    # ── Step 2: images ────────────────────────────────────────────────────────
    if raw_type == "image":
        intent.doc_type          = "image"
        intent.source            = "image"
        intent.chunking_strategy = "image_caption"
        intent.embed_model       = "clip"
        intent.named_vector      = "image"
        intent.confidence        = 0.99
        intent.detection_method  = "magic+extension"
        intent.reason            = "Binary image file"
        return intent

    # ── Step 3: HTML ──────────────────────────────────────────────────────────
    if raw_type == "html":
        intent.doc_type          = "html"
        intent.source            = "html"
        intent.chunking_strategy = "recursive"
        intent.embed_model       = "small"
        intent.named_vector      = "text_small"
        intent.content_sample    = _peek_html(file_bytes)
        intent.confidence        = 0.95
        intent.detection_method  = "magic+extension"
        intent.reason            = "HTML/XML page"
        return intent

    # ── Step 4: PDF / document ────────────────────────────────────────────────
    if raw_type == "pdf":
        pdf_info = _peek_pdf(file_bytes)
        sample   = pdf_info["text_sample"]
        total    = pdf_info["total_pages"]
        tables   = pdf_info["table_page_count"]
        density  = pdf_info["avg_text_density"]
        sampled  = pdf_info["sample_page_count"]

        intent.source         = "pdf"
        intent.content_sample = sample
        intent.is_scanned     = (density < 50 and total > 0)

        table_ratio = tables / max(1, sampled)
        intent.is_table_heavy = table_ratio >= 0.3

        # Legal detection: filename keywords OR content keywords
        legal_content_kws = _LEGAL_KEYWORDS | {
            "whereas", "herein", "pursuant", "indemnif", "arbitrat",
            "jurisdiction", "govern", "clause", "section", "exhibit",
            "representation", "warrant",
        }
        sample_lower = sample.lower()
        legal_in_content = sum(1 for kw in legal_content_kws if kw in sample_lower) >= 3
        intent.is_legal = _is_legal_name(filename) or legal_in_content

        if intent.is_legal:
            intent.doc_type = "legal"
        else:
            semantic = _semantic_doc_type(filename, sample)
            intent.doc_type = semantic if semantic else "pdf"

        intent.chunking_strategy = "semantic"
        intent.embed_model   = "large" if intent.is_legal else "small"
        intent.named_vector  = "text_large" if intent.is_legal else "text_small"

        # Confidence: high if we have text, lower if scanned
        intent.confidence = 0.90 if not intent.is_scanned else 0.65
        intent.detection_method = "heuristic+content_peek"
        intent.reason = (
            f"PDF | legal={intent.is_legal} | doc_type={intent.doc_type} "
            f"| table_heavy={intent.is_table_heavy} | scanned={intent.is_scanned} | pages={total}"
        )
        return intent

    # ── Step 5: plain text / other ────────────────────────────────────────────
    try:
        text_sample = file_bytes[:2000].decode("utf-8", errors="replace")
        intent.content_sample = text_sample
    except Exception:
        pass

    intent.doc_type          = "other"
    intent.source            = "other"
    intent.chunking_strategy = "recursive"
    intent.embed_model       = "small"
    intent.named_vector      = "text_small"
    intent.confidence        = 0.70    # uncertain → may trigger LLM pass
    intent.detection_method  = "heuristic"
    intent.reason            = f"Unknown type (ext={Path(filename).suffix})"
    return intent


# ── Pass 2: LLM classification ────────────────────────────────────────────────

_LLM_SYSTEM = """You are a document classifier for a RAG pipeline.
Given a filename and a content sample, output a JSON object (no markdown) with:
{
  "doc_type":           "legal" | "news" | "medical" | "financial" | "research" | "pdf" | "html" | "image" | "other",
  "chunking_strategy":  "semantic" | "recursive" | "image_caption" | "none",
  "embed_model":        "small" | "large" | "clip",
  "named_vector":       "text_small" | "text_large" | "image",
  "is_legal":           true | false,
  "is_table_heavy":     true | false,
  "is_scanned":         true | false,
  "language":           "<ISO-639-1 code, e.g. en>",
  "confidence":         <float 0.0-1.0>,
  "reason":             "<one short sentence>"
}
Rules:
- doc_type=legal + embed_model=large + named_vector=text_large → contracts, NDAs, policies, compliance, statutes, regulations.
- doc_type=news + chunking_strategy=semantic + embed_model=small → news articles, press releases, journalism.
- doc_type=medical + chunking_strategy=semantic + embed_model=small → clinical notes, medical research, patient records, drug info.
- doc_type=financial + chunking_strategy=semantic + embed_model=small → earnings reports, balance sheets, SEC filings, financial statements.
- doc_type=research + chunking_strategy=semantic + embed_model=small → academic papers, scientific studies, whitepapers.
- doc_type=html + chunking_strategy=recursive + embed_model=small → HTML pages, documentation, web content.
- doc_type=image + chunking_strategy=image_caption + embed_model=clip → images, charts, diagrams.
- doc_type=pdf + chunking_strategy=semantic + embed_model=small → general PDFs, reports, manuals.
- doc_type=other + chunking_strategy=recursive + embed_model=small → plain text, CSV, markdown.
Output ONLY the JSON object."""


async def _llm_classify(filename: str, content_sample: str) -> Optional[DocumentIntent]:
    try:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(
            api_key=settings.OPENROUTER_API_KEY,
            base_url=settings.OPENROUTER_BASE_URL,
        )
        user_msg = f"Filename: {filename}\n\nContent sample:\n{content_sample[:1500]}"
        response = await client.chat.completions.create(
            model=settings.ROUTER_MODEL,
            messages=[
                {"role": "system", "content": _LLM_SYSTEM},
                {"role": "user",   "content": user_msg},
            ],
            max_tokens=300,
            temperature=0,
        )
        raw = (response.choices[0].message.content or "{}").strip()
        raw = raw.lstrip("```json").lstrip("```").rstrip("```").strip()
        data = json.loads(raw)

        return DocumentIntent(
            doc_type=data.get("doc_type", "other"),
            source=_source_for(data.get("doc_type", "other")),
            chunking_strategy=data.get("chunking_strategy", "recursive"),
            embed_model=data.get("embed_model", "small"),
            named_vector=data.get("named_vector", "text_small"),
            is_legal=bool(data.get("is_legal", False)),
            is_table_heavy=bool(data.get("is_table_heavy", False)),
            is_scanned=bool(data.get("is_scanned", False)),
            language=str(data.get("language", "en")),
            confidence=float(data.get("confidence", 0.9)),
            detection_method="llm",
            reason=str(data.get("reason", "")),
        )
    except Exception as e:
        logger.warning("[INTENT] LLM classification failed: %s", e)
        return None


def _source_for(doc_type: str) -> str:
    _PDF_TYPES = {"legal", "news", "medical", "financial", "research", "pdf"}
    if doc_type in _PDF_TYPES:
        return "pdf"
    return {"html": "html", "image": "image"}.get(doc_type, "other")


# ── Public API ────────────────────────────────────────────────────────────────

async def detect_intent(
    filename: str,
    file_bytes: bytes,
    *,
    doc_type_hint: Optional[str] = None,
    llm_threshold: float = 0.85,
) -> DocumentIntent:
    """
    Detect document intent from filename + content.

    Pass 1 (always): fast heuristics — extension, magic bytes, content peek.
    Pass 2 (if confidence < llm_threshold): LLM classification for uncertain cases.

    doc_type_hint lets the caller override the final doc_type (e.g. user set doc_type="legal").
    """
    intent = _heuristic_pass(filename, file_bytes)

    _VALID_HINTS = {"legal", "news", "medical", "financial", "research", "pdf", "html", "image", "other"}
    _SEMANTIC_TYPES = {"legal", "news", "medical", "financial", "research", "pdf"}

    # User-provided hint overrides everything
    if doc_type_hint and doc_type_hint in _VALID_HINTS:
        intent.doc_type          = doc_type_hint
        intent.source            = _source_for(doc_type_hint)
        intent.is_legal          = doc_type_hint == "legal"
        intent.embed_model       = "large" if doc_type_hint == "legal" else ("clip" if doc_type_hint == "image" else "small")
        intent.named_vector      = "text_large" if doc_type_hint == "legal" else ("image" if doc_type_hint == "image" else "text_small")
        intent.chunking_strategy = (
            "image_caption" if doc_type_hint == "image"
            else "semantic" if doc_type_hint in _SEMANTIC_TYPES
            else "recursive"
        )
        intent.confidence        = 1.0
        intent.detection_method  = "user_hint"
        intent.reason            = f"User specified doc_type={doc_type_hint}"

        logger.info("[INTENT] User hint applied | doc_type=%s", doc_type_hint)
        return intent

    # LLM pass for uncertain documents
    if intent.confidence < llm_threshold and intent.content_sample:
        logger.info(
            "[INTENT] Low confidence (%.2f), running LLM pass | file=%s",
            intent.confidence, filename,
        )
        llm_result = await _llm_classify(filename, intent.content_sample)
        if llm_result:
            # Carry over the content sample and page stats from heuristic pass
            llm_result.content_sample = intent.content_sample
            llm_result.is_table_heavy = intent.is_table_heavy or llm_result.is_table_heavy
            intent = llm_result

    logger.info(
        "[INTENT] Result | file=%s doc_type=%s embed=%s chunking=%s legal=%s "
        "table_heavy=%s scanned=%s confidence=%.2f method=%s",
        filename, intent.doc_type, intent.embed_model, intent.chunking_strategy,
        intent.is_legal, intent.is_table_heavy, intent.is_scanned,
        intent.confidence, intent.detection_method,
    )
    return intent
