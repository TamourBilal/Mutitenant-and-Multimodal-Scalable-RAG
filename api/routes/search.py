"""
POST /api/v1/search

Pure hybrid search — no LLM. Returns ranked chunks with rerank scores.
"""
from __future__ import annotations

import logging
from typing import Annotated, List, Optional

from fastapi import APIRouter, Depends, HTTPException

logger = logging.getLogger(__name__)
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from agent.query_router import route_query
from api.collection_enum import Collection, doc_type_filter_for, resolve
from db.session import get_db
from embedding.text_embedder import embed_query
from weaviate_store.retriever import SearchResult, hybrid_search

router = APIRouter()


class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    user_id: str = Field(..., description="Your user ID")
    collections: List[Collection] = Field(
        ...,
        description="Collections to search: documents | tables | images",
    )
    k: Optional[int] = Field(None, ge=1, le=50, description="Max chunks — adaptive if omitted")
    alpha: Optional[float] = Field(None, ge=0.0, le=1.0, description="BM25/semantic blend (0=BM25, 1=semantic)")


class ChunkOut(BaseModel):
    rank: int
    content: str
    score: float
    rerank_score: float
    filename: str
    page_no: int
    doc_type: str
    chunk_type: str
    doc_id: str
    date: str
    file_path: str
    collection: str


class QueryPlanOut(BaseModel):
    named_vector: str
    alpha: float
    k: int
    complexity: str
    collections_searched: List[str]
    router_reason: str


class SearchResponse(BaseModel):
    chunks: List[ChunkOut]
    user_id: str
    collections_searched: List[str]
    total_returned: int
    query_plan: QueryPlanOut


@router.post("", response_model=SearchResponse)
async def search_documents(
    body: SearchRequest,
    _db: Annotated[AsyncSession, Depends(get_db)],
) -> SearchResponse:
    """
    Hybrid BM25 + semantic search with cross-encoder reranking. No LLM.

    **Required:**
    - `query` — search text
    - `user_id` — your user ID
    - `collections` — `["documents"]`, `["tables"]`, `["images"]`, or any combination

    `k` is adaptive (router decides) unless overridden.
    Results are ranked by rerank score (cross-encoder).
    """
    if not body.query.strip():
        raise HTTPException(400, "Query cannot be empty")
    if not body.user_id.strip():
        raise HTTPException(400, "user_id is required")

    collection_names = resolve(body.collections)
    auto_doc_type = doc_type_filter_for(body.collections)

    plan = await route_query(body.query, k_override=body.k)
    if body.alpha is not None:
        plan.alpha = body.alpha
    plan.collections = collection_names

    # text_large is only valid for legal documents — force text_small otherwise
    if plan.doc_type_filter != "legal":
        plan.embed_model = "small"
        plan.named_vector = "text_small"

    model = "large" if plan.embed_model == "large" else "small"
    query_vector = await embed_query(body.query, model=model)

    if not query_vector:
        raise HTTPException(500, "Failed to embed query")

    result: SearchResult = await hybrid_search(
        query_text=body.query,
        query_vector=query_vector,
        user_id=body.user_id,
        named_vector=plan.named_vector,
        alpha=plan.alpha,
        k=plan.k,
        score_threshold=0.0,
        doc_type_filter=auto_doc_type,
        rerank=True,
        rerank_top_n=min(plan.k, 10),
        collection_names=collection_names,
    )

    chunks_out = [
        ChunkOut(
            rank=i + 1,
            content=h["content"],
            score=round(float(h.get("score", 0.0)), 4),
            rerank_score=round(float(h.get("rerank_score", 0.0)), 4),
            filename=h.get("filename", ""),
            page_no=int(h.get("page_no") or 0),
            doc_type=h.get("doc_type", ""),
            chunk_type=h.get("chunk_type", "text"),
            doc_id=h.get("doc_id", ""),
            date=str(h.get("date", "")),
            file_path=h.get("file_path", ""),
            collection=h.get("collection", ""),
        )
        for i, h in enumerate(result.hits)
    ]

    return SearchResponse(
        chunks=chunks_out,
        user_id=body.user_id,
        collections_searched=collection_names,
        total_returned=len(chunks_out),
        query_plan=QueryPlanOut(
            named_vector=plan.named_vector,
            alpha=plan.alpha,
            k=plan.k,
            complexity=plan.complexity,
            collections_searched=collection_names,
            router_reason=plan.reason,
        ),
    )
