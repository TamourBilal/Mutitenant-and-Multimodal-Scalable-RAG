from __future__ import annotations

from typing import Annotated, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from agent.answer_gen import AnswerResult, Reference, generate_answer
from agent.query_router import route_query
from api.deps import get_current_user
from db.models import User
from db.session import get_db
from embedding.image_embedder import embed_text_clip
from embedding.text_embedder import embed_query
from weaviate_store.retriever import hybrid_search

router = APIRouter()


# ── Request / Response schemas ────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    doc_type_filter: Optional[str] = None    # restrict to "legal"|"pdf"|"html"|"image"|"other"
    k_override: Optional[int] = None         # override dynamic k (1-50)
    alpha_override: Optional[float] = None   # override alpha (0.0-1.0)


class ReferenceOut(BaseModel):
    citation_index: int
    filename: str
    source: str
    doc_type: str
    page_no: int
    file_path: str
    date: str
    score: float
    chunk_type: str
    doc_id: str
    content_preview: str


class QueryResponse(BaseModel):
    answer: str
    references: List[ReferenceOut]
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
        chunk_type=r.chunk_type,
        doc_id=r.doc_id,
        content_preview=r.content_preview,
    )


# ── Endpoint ──────────────────────────────────────────────────────────────────

@router.post("", response_model=QueryResponse)
async def query_documents(
    body: QueryRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    _db: Annotated[AsyncSession, Depends(get_db)],
) -> QueryResponse:
    """
    Agentic RAG query:
      1. Query router determines retrieval strategy (k, alpha, named_vector).
      2. Hybrid search (BM25 + semantic) in Weaviate with cross-encoder reranking.
      3. Answer generation with inline [n] citations and full references array.
    """
    if not body.query.strip():
        raise HTTPException(400, "Query cannot be empty")

    # Step 1 — route
    plan = await route_query(
        body.query,
        doc_type_hint=body.doc_type_filter,
        k_override=body.k_override,
    )
    if body.alpha_override is not None:
        plan.alpha = max(0.0, min(1.0, body.alpha_override))

    # Step 2 — embed query with the right model
    if plan.embed_model == "clip":
        vecs = await embed_text_clip([body.query])
        query_vector = vecs[0] if vecs else []
    elif plan.embed_model == "large":
        query_vector = await embed_query(body.query, model="large")
    else:
        query_vector = await embed_query(body.query, model="small")

    if not query_vector:
        raise HTTPException(500, "Failed to embed query")

    # Step 3 — hybrid search across router-selected collections
    hits = await hybrid_search(
        query_text=body.query,
        query_vector=query_vector,
        user_id=current_user.id,
        named_vector=plan.named_vector,
        alpha=plan.alpha,
        k=plan.k,
        doc_type_filter=plan.doc_type_filter,
        rerank=True,
        rerank_top_n=min(plan.k, 8),
        collection_names=plan.collections,
    )

    # Step 4 — generate answer
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
        k_used=plan.k,
        alpha_used=plan.alpha,
        named_vector_used=plan.named_vector,
        tokens_used=result.tokens_used,
        router_reason=plan.reason,
    )
