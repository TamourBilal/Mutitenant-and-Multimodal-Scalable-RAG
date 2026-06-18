"""
POST /api/v1/search

Pure semantic search — no LLM involved.
Returns ranked chunks with full source metadata so callers can inspect
the retrieval quality before committing to an answer-generation call.
"""
from __future__ import annotations

from typing import Annotated, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from agent.query_router import route_query
from api.deps import get_current_user
from config import settings
from db.models import User
from db.session import get_db
from embedding.image_embedder import embed_text_clip
from embedding.text_embedder import embed_query
from weaviate_store.retriever import SearchResult, hybrid_search

router = APIRouter()


# ── Request / Response schemas ────────────────────────────────────────────────

class SearchRequest(BaseModel):
    query: str
    doc_type_filter: Optional[str] = Field(
        None,
        description="Restrict search to a doc_type: legal | news | medical | financial | research | pdf | html | image",
    )
    k: Optional[int] = Field(None, ge=1, le=50, description="Max chunks to return (overrides dynamic k)")
    alpha: Optional[float] = Field(None, ge=0.0, le=1.0, description="BM25/semantic blend (0=BM25, 1=semantic)")
    score_threshold: Optional[float] = Field(
        None, ge=0.0, le=1.0,
        description="Minimum Weaviate hybrid score to include a chunk (default from config)",
    )


class ChunkOut(BaseModel):
    rank: int
    content: str
    score: float            # Weaviate hybrid score
    rerank_score: float     # cross-encoder score (higher = more relevant)
    filename: str
    page_no: int
    doc_type: str
    chunk_type: str
    doc_id: str
    date: str
    file_path: str
    collection: str         # RAGDocuments | RAGTables | RAGImages


class QueryPlanOut(BaseModel):
    named_vector: str
    alpha: float
    k: int
    score_threshold: float
    collections: List[str]
    doc_type_filter: Optional[str]
    router_reason: str


class SearchResponse(BaseModel):
    chunks: List[ChunkOut]
    total_returned: int
    total_filtered: int       # chunks dropped below score_threshold
    score_threshold: float
    query_plan: QueryPlanOut


# ── Endpoint ──────────────────────────────────────────────────────────────────

@router.post("", response_model=SearchResponse)
async def search_documents(
    body: SearchRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    _db: Annotated[AsyncSession, Depends(get_db)],
) -> SearchResponse:
    """
    Hybrid semantic + BM25 search over the user's document corpus.

    Returns ranked chunks with source metadata. No LLM call is made — use
    `/ask` for a generated answer with citations.

    **Score threshold**: chunks below the threshold are silently dropped and
    counted in `total_filtered`. Raise the threshold for higher precision;
    lower it for higher recall.
    """
    if not body.query.strip():
        raise HTTPException(400, "Query cannot be empty")

    # Route the query to determine retrieval strategy
    plan = await route_query(
        body.query,
        doc_type_hint=body.doc_type_filter,
        k_override=body.k,
        threshold_override=body.score_threshold,
    )
    if body.alpha is not None:
        plan.alpha = body.alpha

    # Embed query with the right model
    if plan.embed_model == "clip":
        vecs = await embed_text_clip([body.query])
        query_vector = vecs[0] if vecs else []
    elif plan.embed_model == "large":
        query_vector = await embed_query(body.query, model="large")
    else:
        query_vector = await embed_query(body.query, model="small")

    if not query_vector:
        raise HTTPException(500, "Failed to embed query")

    # Hybrid search with threshold gate
    result: SearchResult = await hybrid_search(
        query_text=body.query,
        query_vector=query_vector,
        user_id=current_user.id,
        named_vector=plan.named_vector,
        alpha=plan.alpha,
        k=plan.k,
        doc_type_filter=plan.doc_type_filter,
        score_threshold=plan.score_threshold,
        rerank=True,
        rerank_top_n=min(plan.k, 10),
        collection_names=plan.collections,
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
        total_returned=len(chunks_out),
        total_filtered=result.filtered_count,
        score_threshold=result.score_threshold,
        query_plan=QueryPlanOut(
            named_vector=plan.named_vector,
            alpha=plan.alpha,
            k=plan.k,
            score_threshold=result.score_threshold,
            collections=plan.collections,
            doc_type_filter=plan.doc_type_filter,
            router_reason=plan.reason,
        ),
    )
