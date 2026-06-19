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

NamedVector = Literal["text_small", "text_large"]

_ROUTER_SYSTEM = f"""You are a retrieval strategy planner for a RAG system with THREE Weaviate collections:
  - "{settings.WEAVIATE_COLLECTION_DOCUMENTS}" — text chunks from PDFs and HTML pages
  - "{settings.WEAVIATE_COLLECTION_TABLES}"    — table chunks extracted from PDFs
  - "{settings.WEAVIATE_COLLECTION_IMAGES}"    — image caption chunks

Given a user query, output a JSON object (no markdown, pure JSON) with:

{{
  "named_vector":    "text_small" | "text_large",
  "alpha":           <float 0.0–1.0>,
  "k":               <int>,
  "complexity":      "simple" | "moderate" | "complex",
  "score_threshold": <float 0.0–1.0>,
  "doc_type_filter": <string or null>,
  "embed_model":     "small" | "large",
  "collections":     [<list of collection names to search>],
  "reason":          <short string>
}}

── K SIZING (based on query complexity) ──────────────────────────────────────
complexity="simple"   → k = 5–10
  Single fact, single metric, single entity.
  Examples: "What is Africa GDP?", "Population of Eastern Africa?"

complexity="moderate" → k = 10–20
  Comparison of 2-3 entities, explanation, multi-part.
  Examples: "Compare GDP of Northern vs Southern Africa"

complexity="complex"  → k = 25–40
  "all", "every", "summarize all", broad analysis, tables needed.
  Examples: "Tell me GDP of all countries",
            "Summarize all regional statistics with a table",
            "Compare all regions", "give me everything about X"
  ANY query with "all", "every", "summarize", "overview" → complex, k=25+

── OTHER RULES ────────────────────────────────────────────────────────────────
- named_vector="text_large", embed_model="large" → ONLY for legal/contract/compliance documents
- named_vector="text_small", embed_model="small" → ALL other queries (stats, finance, research, news, images, general)
- image/visual queries → collections=["{settings.WEAVIATE_COLLECTION_IMAGES}"]
- table/statistics queries → include "{settings.WEAVIATE_COLLECTION_TABLES}"
- broad/compare queries → include all three collections
- alpha: 0.3=keyword, 0.75=semantic, 0.5=mixed (use 0.5 for stats/numbers)
- score_threshold: simple=0.75, moderate=0.65, complex=0.55

Output ONLY the JSON object."""


@dataclass
class QueryPlan:
    named_vector: NamedVector = "text_small"
    alpha: float = 0.75
    k: int = 10
    complexity: str = "moderate"       # simple | moderate | complex
    score_threshold: float = settings.SCORE_THRESHOLD
    doc_type_filter: Optional[str] = None
    embed_model: Literal["small", "large"] = "small"
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

        complexity = data.get("complexity", "moderate")

        # Enforce k bounds per complexity tier
        raw_k = int(data.get("k", 10))
        if complexity == "simple":
            k = max(5, min(raw_k, 10))
        elif complexity == "complex":
            k = max(25, min(raw_k, 40))
        else:  # moderate
            k = max(10, min(raw_k, 20))

        plan = QueryPlan(
            named_vector=data.get("named_vector", "text_small"),
            alpha=float(data.get("alpha", 0.75)),
            k=k,
            complexity=complexity,
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
            "[ROUTER] Plan | complexity=%s k=%d nv=%s alpha=%.2f threshold=%.2f "
            "collections=%s reason=%s",
            plan.complexity, plan.k, plan.named_vector, plan.alpha,
            plan.score_threshold, plan.collections, plan.reason,
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
