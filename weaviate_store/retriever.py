from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from config import settings
from weaviate_store.client import get_weaviate_client
from weaviate_store.schema import COLLECTION_VECTORS

logger = logging.getLogger(__name__)

_reranker = None


def _get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        logger.info("Cross-encoder reranker loaded")
    return _reranker


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
    if named_vector not in supported:
        logger.debug(
            "[RETRIEVER] Skipping collection — vector not supported | "
            "collection=%s named_vector=%s",
            collection_name, named_vector,
        )
        return []

    client = get_weaviate_client()
    collection = client.collections.get(collection_name)
    tenant_col = collection.with_tenant(user_id)

    filters = None
    if doc_type_filter:
        filters = Filter.by_property("doc_type").equal(doc_type_filter)

    result = tenant_col.query.hybrid(
        query=query_text,
        vector=query_vector,
        target_vector=named_vector,
        alpha=alpha,
        limit=k,
        filters=filters,
        return_metadata=MetadataQuery(score=True, distance=True),
        return_properties=[
            "content", "source", "doc_type", "date",
            "doc_id", "chunk_index", "page_no",
            "file_path", "filename", "chunk_type", "user_id",
        ],
    )

    hits = []
    for obj in result.objects:
        props = dict(obj.properties)
        hits.append(
            {
                "id":          str(obj.uuid),
                "score":       obj.metadata.score if obj.metadata else 0.0,
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
        )
    return hits


def _rerank(query: str, hits: List[Dict[str, Any]], top_n: int) -> List[Dict[str, Any]]:
    if not hits:
        return hits
    reranker = _get_reranker()
    pairs = [(query, h["content"]) for h in hits]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(hits, scores), key=lambda x: x[1], reverse=True)
    for hit, score in ranked:
        hit["rerank_score"] = float(score)
    return [h for h, _ in ranked[:top_n]]


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

    tasks = [
        asyncio.to_thread(
            _sync_hybrid_search,
            query_text=query_text,
            query_vector=query_vector,
            user_id=user_id,
            named_vector=named_vector,
            alpha=alpha,
            k=k,
            doc_type_filter=doc_type_filter,
            collection_name=coll,
        )
        for coll in collection_names
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    all_hits: List[Dict[str, Any]] = []
    for coll, result in zip(collection_names, results):
        if isinstance(result, Exception):
            logger.error("[RETRIEVER] Search failed | collection=%s err=%s", coll, result)
        else:
            all_hits.extend(result)

    # Merge: sort by hybrid score descending, cap at k
    all_hits.sort(key=lambda h: h.get("score", 0.0), reverse=True)
    all_hits = all_hits[:k]
    total_fetched = len(all_hits)

    # ── Score threshold gate ─────────────────────────────────────────────────
    filtered_count = 0
    effective_threshold = score_threshold if score_threshold > 0 else settings.SCORE_THRESHOLD
    if effective_threshold > 0:
        passing = [h for h in all_hits if h.get("score", 0.0) >= effective_threshold]
        filtered_count = total_fetched - len(passing)
        if filtered_count:
            logger.info(
                "[RETRIEVER] Threshold gate | threshold=%.2f removed=%d remaining=%d",
                effective_threshold, filtered_count, len(passing),
            )
        all_hits = passing

    logger.info(
        "[RETRIEVER] Search done | collections=%s nv=%s alpha=%.2f k=%d "
        "fetched=%d filtered=%d remaining=%d user=%s",
        collection_names, named_vector, alpha, k,
        total_fetched, filtered_count, len(all_hits), user_id,
    )

    # ── Cross-encoder rerank ─────────────────────────────────────────────────
    if rerank and len(all_hits) > 1:
        all_hits = await asyncio.to_thread(_rerank, query_text, all_hits, rerank_top_n)
        logger.info("[RETRIEVER] Reranked | top_n=%d", len(all_hits))

    return SearchResult(
        hits=all_hits,
        total_fetched=total_fetched,
        filtered_count=filtered_count,
        score_threshold=effective_threshold,
    )
