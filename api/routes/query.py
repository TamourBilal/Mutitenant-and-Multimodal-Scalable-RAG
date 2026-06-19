from __future__ import annotations

from typing import Annotated, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from agent.answer_gen import AnswerResult, Reference, generate_answer
from agent.query_router import route_query
from api.collection_enum import Collection, resolve
from db.session import get_db
from embedding.text_embedder import embed_query
from weaviate_store.retriever import hybrid_search

router = APIRouter()


class QueryRequest(BaseModel):
    query: str = Field(..., description="The question to answer")
    user_id: str = Field(..., description="Your user ID")
    collections: List[Collection] = Field(
        ...,
        description="Collections to search: documents | tables | images",
    )
    k_override: Optional[int] = Field(None, ge=1, le=50, description="Max chunks — adaptive if omitted")
    alpha_override: Optional[float] = Field(None, ge=0.0, le=1.0, description="BM25/semantic blend")


class ReferenceOut(BaseModel):
    citation_index: int
    filename: str
    source: str
    doc_type: str
    page_no: int
    file_path: str
    date: str
    score: float
    rerank_score: float
    chunk_type: str
    doc_id: str
    content_preview: str


class QueryResponse(BaseModel):
    answer: str
    references: List[ReferenceOut]
    user_id: str
    collections_searched: List[str]
    k_used: int
    alpha_used: float
    named_vector_used: str
    tokens_used: int
    router_reason: str


def _ref_to_out(r: Reference) -> ReferenceOut:
    return ReferenceOut(
        citation_index=r.citation_index,
        filename=r.filename,
        source=r.source,
        doc_type=r.doc_type,
        page_no=r.page_no,
        file_path=r.file_path,
        date=r.date,
        score=r.score,
        rerank_score=getattr(r, "rerank_score", 0.0),
        chunk_type=r.chunk_type,
        doc_id=r.doc_id,
        content_preview=r.content_preview,
    )


@router.post("", response_model=QueryResponse)
async def query_documents(
    body: QueryRequest,
    _db: Annotated[AsyncSession, Depends(get_db)],
) -> QueryResponse:
    """
    Agentic RAG query with answer generation and inline citations.

    **Required:**
    - `query` — what you want to ask
    - `user_id` — your user ID
    - `collections` — `["documents"]`, `["tables"]`, `["images"]`, or any combination

    `k_override` is optional — router decides adaptively if omitted.
    Relevance threshold is calculated internally from rerank scores.
    """
    if not body.query.strip():
        raise HTTPException(400, "Query cannot be empty")
    if not body.user_id.strip():
        raise HTTPException(400, "user_id is required")

    collection_names = resolve(body.collections)

    plan = await route_query(body.query, k_override=body.k_override)
    if body.alpha_override is not None:
        plan.alpha = max(0.0, min(1.0, body.alpha_override))
    plan.collections = collection_names

    if plan.doc_type_filter != "legal":
        plan.embed_model = "small"
        plan.named_vector = "text_small"

    model = "large" if plan.embed_model == "large" else "small"
    query_vector = await embed_query(body.query, model=model)

    if not query_vector:
        raise HTTPException(500, "Failed to embed query")

    hits = await hybrid_search(
        query_text=body.query,
        query_vector=query_vector,
        user_id=body.user_id,
        named_vector=plan.named_vector,
        alpha=plan.alpha,
        k=plan.k,
        score_threshold=0.0,   # threshold derived from rerank internally
        rerank=True,
        rerank_top_n=min(plan.k, 8),
        collection_names=collection_names,
    )

    result: AnswerResult = await generate_answer(
        body.query,
        hits,
        k_used=plan.k,
        alpha_used=plan.alpha,
        named_vector_used=plan.named_vector,
    )

    return QueryResponse(
        answer=result.answer,
        references=[_ref_to_out(r) for r in result.references],
        user_id=body.user_id,
        collections_searched=collection_names,
        k_used=plan.k,
        alpha_used=plan.alpha,
        named_vector_used=plan.named_vector,
        tokens_used=result.tokens_used,
        router_reason=plan.reason,
    )
