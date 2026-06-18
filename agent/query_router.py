"""
Agentic query router: calls OpenRouter to produce a structured retrieval plan
controlling which Weaviate collections to search, the named vector, alpha
(BM25 vs semantic blend), dynamic top-k, and score threshold.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import List, Literal, Optional

from openai import AsyncOpenAI

from config import settings

logger = logging.getLogger(__name__)

NamedVector = Literal["text_small", "text_large", "image"]

_ROUTER_SYSTEM = f"""You are a retrieval strategy planner for a RAG system with THREE Weaviate collections:
  - "{settings.WEAVIATE_COLLECTION_DOCUMENTS}" — text chunks from PDFs and HTML pages
  - "{settings.WEAVIATE_COLLECTION_TABLES}"    — table chunks extracted from PDFs
  - "{settings.WEAVIATE_COLLECTION_IMAGES}"    — image caption chunks (CLIP + text)

Given a user query, output a JSON object (no markdown, pure JSON) with:

{{
  "named_vector":    "text_small" | "text_large" | "image",
  "alpha":           <float 0.0–1.0>,
  "k":               <int 3–30>,
  "score_threshold": <float 0.0–1.0>,
  "doc_type_filter": <string or null>,
  "embed_model":     "small" | "large" | "clip",
  "collections":     [<list of collection names to search>],
  "reason":          <short string>
}}

Rules:
- named_vector="text_large", embed_model="large"  → legal/contract/compliance queries;
  collections=["{settings.WEAVIATE_COLLECTION_DOCUMENTS}", "{settings.WEAVIATE_COLLECTION_TABLES}"]
- named_vector="image", embed_model="clip"        → visual/diagram/image queries;
  collections=["{settings.WEAVIATE_COLLECTION_IMAGES}"]
- named_vector="text_small", embed_model="small"  → all other queries;
  default collections=["{settings.WEAVIATE_COLLECTION_DOCUMENTS}", "{settings.WEAVIATE_COLLECTION_TABLES}"]
- For broad summarise/compare queries spanning all content: include all three collections.
- For table-specific queries (numbers, statistics): add "{settings.WEAVIATE_COLLECTION_TABLES}".

score_threshold:
- Simple factual queries → 0.75 (high bar, precision over recall)
- Complex/multi-part queries → 0.60 (lower bar, recall over precision)
- Default → 0.70

alpha: 0.3 for keyword/exact-match; 0.75 for conceptual/semantic; 0.5 for mixed.
k: 3 for simple facts, 10–15 for explanations, 20–30 for summarise/compare.
doc_type_filter: "legal" | "news" | "medical" | "financial" | "research" | "pdf" | "html" | "image" | null.
Output ONLY the JSON object."""


@dataclass
class QueryPlan:
    named_vector: NamedVector = "text_small"
    alpha: float = 0.75
    k: int = 10
    score_threshold: float = settings.SCORE_THRESHOLD
    doc_type_filter: Optional[str] = None
    embed_model: Literal["small", "large", "clip"] = "small"
    reason: str = ""
    collections: List[str] = field(
        default_factory=lambda: [
            settings.WEAVIATE_COLLECTION_DOCUMENTS,
            settings.WEAVIATE_COLLECTION_TABLES,
        ]
    )


_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(
            api_key=settings.OPENROUTER_API_KEY,
            base_url=settings.OPENROUTER_BASE_URL,
        )
    return _client


async def route_query(
    query: str,
    *,
    doc_type_hint: Optional[str] = None,
    k_override: Optional[int] = None,
    threshold_override: Optional[float] = None,
) -> QueryPlan:
    """
    Call the LLM router to determine retrieval strategy.
    Falls back to sensible defaults on any error.
    """
    user_msg = query
    if doc_type_hint:
        user_msg = f"[User wants to search {doc_type_hint} documents]\n{query}"

    try:
        client = _get_client()
        response = await client.chat.completions.create(
            model=settings.ROUTER_MODEL,
            messages=[
                {"role": "system", "content": _ROUTER_SYSTEM},
                {"role": "user",   "content": user_msg},
            ],
            max_tokens=250,
            temperature=0,
        )
        raw = response.choices[0].message.content or "{}"
        raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        data = json.loads(raw)

        valid_colls = {
            settings.WEAVIATE_COLLECTION_DOCUMENTS,
            settings.WEAVIATE_COLLECTION_TABLES,
            settings.WEAVIATE_COLLECTION_IMAGES,
        }
        colls = [c for c in (data.get("collections") or []) if c in valid_colls]
        if not colls:
            colls = [settings.WEAVIATE_COLLECTION_DOCUMENTS, settings.WEAVIATE_COLLECTION_TABLES]

        threshold = float(data.get("score_threshold", settings.SCORE_THRESHOLD))
        threshold = max(0.0, min(1.0, threshold))

        plan = QueryPlan(
            named_vector=data.get("named_vector", "text_small"),
            alpha=float(data.get("alpha", 0.75)),
            k=int(data.get("k", 10)),
            score_threshold=threshold,
            doc_type_filter=data.get("doc_type_filter") or doc_type_hint,
            embed_model=data.get("embed_model", "small"),
            reason=data.get("reason", ""),
            collections=colls,
        )

        if k_override:
            plan.k = k_override
        if threshold_override is not None:
            plan.score_threshold = max(0.0, min(1.0, threshold_override))

        logger.info(
            "[ROUTER] Plan | nv=%s alpha=%.2f k=%d threshold=%.2f "
            "collections=%s filter=%s reason=%s",
            plan.named_vector, plan.alpha, plan.k, plan.score_threshold,
            plan.collections, plan.doc_type_filter, plan.reason,
        )
        return plan

    except Exception as exc:
        logger.warning("[ROUTER] Fallback to defaults | err=%s", exc)
        plan = QueryPlan(
            doc_type_filter=doc_type_hint,
            score_threshold=threshold_override if threshold_override is not None else settings.SCORE_THRESHOLD,
        )
        if k_override:
            plan.k = k_override
        return plan
