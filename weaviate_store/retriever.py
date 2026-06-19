from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from config import settings
from weaviate_store.client import get_weaviate_client
from weaviate_store.schema import COLLECTION_VECTORS

logger = logging.getLogger(__name__)

_RERANK_SYSTEM = """You are a relevance reranker for a RAG system.
Given a query and a list of retrieved chunks, score each chunk for relevance.

Output ONLY a JSON array of objects in this exact format (no markdown):
[{"index": 0, "score": 0.95}, {"index": 1, "score": 0.42}, ...]

Scoring rules:
- 0.9–1.0: directly answers the query with specific facts
- 0.7–0.9: strongly relevant, contains related information
- 0.4–0.7: partially relevant, tangentially related
- 0.0–0.4: not relevant to the query

Return scores for ALL chunks. Output ONLY the JSON array."""


@dataclass
class SearchResult:
    """Return value of hybrid_search — hits that passed the threshold plus audit counts."""
    hits: List[Dict[str, Any]] = field(default_factory=list)
    total_fetched: int = 0       # results returned by Weaviate before threshold
    filtered_count: int = 0      # chunks dropped because score < score_threshold
    score_threshold: float = 0.0


def _sync_hybrid_search(
    *,
    query_text: str,
    query_vector: List[float],
    user_id: str,
    named_vector: str,
    alpha: float,
    k: int,
    doc_type_filter: Optional[str],
    collection_name: str,
) -> List[Dict[str, Any]]:
    from weaviate.classes.query import Filter, MetadataQuery

    supported = COLLECTION_VECTORS.get(collection_name, [])
    logger.info(
        "[RETRIEVER] Searching | collection=%s named_vector=%s supported=%s alpha=%.2f k=%d user=%s",
        collection_name, named_vector, supported, alpha, k, user_id,
    )
    if named_vector not in supported:
        logger.warning(
            "[RETRIEVER] Skipping — vector not in collection | collection=%s named_vector=%s supported=%s",
            collection_name, named_vector, supported,
        )
        return []

    client = get_weaviate_client()
    collection = client.collections.get(collection_name)
    tenant_col = collection.with_tenant(user_id)

    filters = None
    if doc_type_filter:
        filters = Filter.by_property("doc_type").equal(doc_type_filter)

    _RETURN_PROPS = [
        "content", "source", "doc_type", "date",
        "doc_id", "chunk_index", "page_no",
        "file_path", "filename", "chunk_type", "user_id",
    ]

    # ── BM25 sanity check: confirm tenant has data ────────────────────────────
    bm25_check = tenant_col.query.bm25(
        query=query_text,
        limit=3,
        return_properties=_RETURN_PROPS,
    )
    logger.info(
        "[RETRIEVER] BM25 sanity | collection=%s user=%s bm25_hits=%d",
        collection_name, user_id, len(bm25_check.objects),
    )

    # ── Hybrid search (BM25 + vector) ─────────────────────────────────────────
    # Use alpha=1.0 (pure semantic) fallback if hybrid returns nothing
    hybrid_result = tenant_col.query.hybrid(
        query=query_text,
        vector=query_vector,
        target_vector=named_vector,
        alpha=alpha,
        limit=k,
        filters=filters,
        return_metadata=MetadataQuery(score=True, distance=True),
        return_properties=_RETURN_PROPS,
    )
    logger.info(
        "[RETRIEVER] Hybrid result | collection=%s named_vector=%s alpha=%.2f hits=%d",
        collection_name, named_vector, alpha, len(hybrid_result.objects),
    )

    # Fallback to BM25-only if hybrid returned nothing (vector dim mismatch guard)
    if not hybrid_result.objects and bm25_check.objects:
        logger.warning(
            "[RETRIEVER] Hybrid returned 0 but BM25 has data — falling back to BM25 | collection=%s",
            collection_name,
        )
        fallback = tenant_col.query.bm25(
            query=query_text,
            limit=k,
            return_properties=_RETURN_PROPS,
        )
        source_objects = fallback.objects
        use_bm25_score = True
    else:
        source_objects = hybrid_result.objects
        use_bm25_score = False

    def _build_hit(obj, use_bm25: bool) -> dict:
        props = dict(obj.properties)
        score = 0.5 if use_bm25 else (obj.metadata.score if obj.metadata else 0.0)
        return {
            "id":          str(obj.uuid),
            "score":       score,
            "content":     props.get("content", ""),
            "source":      props.get("source", ""),
            "doc_type":    props.get("doc_type", ""),
            "date":        str(props.get("date", "")),
            "doc_id":      props.get("doc_id", ""),
            "chunk_index": props.get("chunk_index", 0),
            "page_no":     props.get("page_no", 0),
            "file_path":   props.get("file_path", ""),
            "filename":    props.get("filename", ""),
            "chunk_type":  props.get("chunk_type", "text"),
            "collection":  collection_name,
        }

    return [_build_hit(obj, use_bm25_score) for obj in source_objects]


async def _rerank(query: str, hits: List[Dict[str, Any]], top_n: int) -> List[Dict[str, Any]]:
    """Rerank hits using OpenRouter RERANK_MODEL (gpt-4o-mini by default)."""
    if not hits:
        return hits

    import json
    from openai import AsyncOpenAI

    # Build numbered chunk list for the prompt
    chunks_text = "\n\n".join(
        f"[{i}] {h['content'][:400]}" for i, h in enumerate(hits)
    )
    user_msg = f"Query: {query}\n\nChunks:\n{chunks_text}"

    try:
        client = AsyncOpenAI(
            api_key=settings.OPENROUTER_API_KEY,
            base_url=settings.OPENROUTER_BASE_URL,
        )
        response = await client.chat.completions.create(
            model=settings.RERANK_MODEL,
            messages=[
                {"role": "system", "content": _RERANK_SYSTEM},
                {"role": "user",   "content": user_msg},
            ],
            max_tokens=300,
            temperature=0,
        )
        raw = (response.choices[0].message.content or "[]").strip()
        raw = raw.lstrip("```json").lstrip("```").rstrip("```").strip()
        scores_data = json.loads(raw)

        # Apply scores back to hits
        score_map = {item["index"]: float(item["score"]) for item in scores_data}
        for i, hit in enumerate(hits):
            hit["rerank_score"] = score_map.get(i, 0.0)

        ranked = sorted(hits, key=lambda h: h.get("rerank_score", 0.0), reverse=True)
        logger.info(
            "[RETRIEVER] Reranked via %s | chunks=%d top_n=%d",
            settings.RERANK_MODEL, len(hits), top_n,
        )
        return ranked[:top_n]

    except Exception as exc:
        logger.warning("[RETRIEVER] Rerank failed, returning original order | err=%s", exc)
        for hit in hits:
            hit.setdefault("rerank_score", hit.get("score", 0.0))
        return hits[:top_n]


async def hybrid_search(
    *,
    query_text: str,
    query_vector: List[float],
    user_id: str,
    named_vector: str = "text_small",
    alpha: float = 0.75,
    k: int = 10,
    doc_type_filter: Optional[str] = None,
    score_threshold: float = 0.0,
    rerank: bool = True,
    rerank_top_n: int = 6,
    rerank_threshold: float = 0.5,   # drop chunks below this rerank score before LLM
    collection_names: Optional[List[str]] = None,
) -> SearchResult:
    """
    Parallel hybrid BM25 + semantic search across one or more Weaviate collections.

    Threshold filtering:
      Chunks whose Weaviate hybrid score < score_threshold are removed BEFORE
      reranking.  This keeps the cross-encoder's workload small and ensures the
      LLM only sees chunks that cleared the relevance bar.

    Returns a SearchResult with hits, total_fetched, filtered_count, score_threshold.
    """
    if collection_names is None:
        collection_names = [
            settings.WEAVIATE_COLLECTION_DOCUMENTS,
            settings.WEAVIATE_COLLECTION_TABLES,
        ]

    async def _run_search(nv: str) -> List[Dict[str, Any]]:
        tasks = [
            asyncio.to_thread(
                _sync_hybrid_search,
                query_text=query_text,
                query_vector=query_vector,
                user_id=user_id,
                named_vector=nv,
                alpha=alpha,
                k=k,
                doc_type_filter=doc_type_filter,
                collection_name=coll,
            )
            for coll in collection_names
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        hits: List[Dict[str, Any]] = []
        for coll, result in zip(collection_names, results):
            if isinstance(result, Exception):
                logger.error("[RETRIEVER] Search failed | collection=%s err=%s", coll, result)
            else:
                hits.extend(result)
        return hits

    all_hits = await _run_search(named_vector)

    # Fallback: text_large returned nothing → retry with text_small
    if not all_hits and named_vector == "text_large":
        logger.info("[RETRIEVER] text_large returned 0 hits — retrying with text_small")
        named_vector = "text_small"
        all_hits = await _run_search("text_small")

    # Deduplicate by (page_no, content[:80]) — removes duplicate ingestion runs
    seen_keys = set()
    deduped = []
    for h in all_hits:
        key = (h.get("page_no"), h.get("content", "")[:80])
        if key not in seen_keys:
            seen_keys.add(key)
            deduped.append(h)
    all_hits = deduped
    logger.info("[RETRIEVER] After dedup | hits=%d (removed %d duplicates)", len(all_hits), len(deduped) - len(all_hits) + (len(all_hits) - len(deduped)))

    # Merge: sort by hybrid score descending, cap at k
    all_hits.sort(key=lambda h: h.get("score", 0.0), reverse=True)
    all_hits = all_hits[:k]
    total_fetched = len(all_hits)

    logger.info(
        "[RETRIEVER] Raw hits | collections=%s nv=%s alpha=%.2f k=%d fetched=%d user=%s",
        collection_names, named_vector, alpha, k, total_fetched, user_id,
    )
    for i, h in enumerate(all_hits):
        logger.info(
            "[RETRIEVER] hit[%d] score=%.4f page=%s doc=%s content_preview=%s",
            i, h.get("score", 0.0), h.get("page_no"), h.get("doc_id", "")[:8],
            h.get("content", "")[:60].replace("\n", " "),
        )

    # ── Score threshold gate (only applied when explicitly set > 0) ──────────
    filtered_count = 0
    effective_threshold = score_threshold if score_threshold > 0 else 0.0
    if effective_threshold > 0:
        passing = [h for h in all_hits if h.get("score", 0.0) >= effective_threshold]
        filtered_count = total_fetched - len(passing)
        logger.info(
            "[RETRIEVER] Threshold gate | threshold=%.2f removed=%d remaining=%d",
            effective_threshold, filtered_count, len(passing),
        )
        all_hits = passing
    else:
        logger.info("[RETRIEVER] No threshold gate — returning all %d hits", total_fetched)

    logger.info(
        "[RETRIEVER] After gate | remaining=%d filtered=%d",
        len(all_hits), filtered_count,
    )

    # ── LLM rerank via OpenRouter ────────────────────────────────────────────
    if rerank and len(all_hits) > 1:
        all_hits = await _rerank(query_text, all_hits, rerank_top_n)

        # ── Rerank threshold gate — only high-confidence chunks reach the LLM ──
        before_rerank_gate = len(all_hits)
        all_hits = [h for h in all_hits if h.get("rerank_score", 0.0) >= rerank_threshold]
        rerank_filtered = before_rerank_gate - len(all_hits)
        filtered_count += rerank_filtered
        logger.info(
            "[RETRIEVER] Rerank threshold gate | threshold=%.2f removed=%d remaining=%d",
            rerank_threshold, rerank_filtered, len(all_hits),
        )

    logger.info("[RETRIEVER] Final chunks for LLM | count=%d", len(all_hits))

    return SearchResult(
        hits=all_hits,
        total_fetched=total_fetched,
        filtered_count=filtered_count,
        score_threshold=effective_threshold,
    )
