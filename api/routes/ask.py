"""
POST /api/v1/ask

Full RAG: hybrid search → cross-encoder rerank → LLM answer with citations.
score_threshold is calculated internally from rerank scores.
"""
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
from weaviate_store.retriever import SearchResult, hybrid_search

router = APIRouter()

_LOW_CONFIDENCE_MSG = (
    "I could not find relevant chunks for your question. "
    "Try rephrasing or ingesting more relevant documents."
)


class AskRequest(BaseModel):
    question: str = Field(..., description="The question to answer")
    user_id: str = Field(..., description="Your user ID")
    collections: List[Collection] = Field(
        ...,
        description="Collections to search: documents | tables | images",
    )
    k: Optional[int] = Field(None, ge=1, le=50, description="Max chunks — adaptive if omitted")
    alpha: Optional[float] = Field(None, ge=0.0, le=1.0, description="BM25/semantic blend (0=BM25, 1=semantic)")


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


class AskResponse(BaseModel):
    answer: str
    references: List[ReferenceOut]
    user_id: str
    collections_searched: List[str]
    chunks_used: int
    chunks_filtered: int
    k_used: int
    alpha_used: float
    named_vector_used: str
    tokens_used: int
    router_reason: str
    low_confidence: bool
    guardrail_passed: bool
    guardrail_verdict: str
    query_complexity: str


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


@router.post("", response_model=AskResponse)
async def ask_question(
    body: AskRequest,
    _db: Annotated[AsyncSession, Depends(get_db)],
) -> AskResponse:
    """
    Full RAG pipeline: retrieve → rerank → LLM answer.

    **Required:**
    - `question` — what you want to ask
    - `user_id` — your user ID
    - `collections` — which to search: `["documents"]`, `["tables"]`, `["images"]`

    `k` is adaptive (router decides) unless overridden.
    Relevance threshold is calculated internally from rerank scores.
    """
    if not body.question.strip():
        raise HTTPException(400, "Question cannot be empty")
    if not body.user_id.strip():
        raise HTTPException(400, "user_id is required")

    collection_names = resolve(body.collections)

    # Step 1 — adaptive routing (k, alpha, named_vector)
    plan = await route_query(body.question, k_override=body.k)
    if body.alpha is not None:
        plan.alpha = max(0.0, min(1.0, body.alpha))
    plan.collections = collection_names

    if plan.doc_type_filter != "legal":
        plan.embed_model = "small"
        plan.named_vector = "text_small"

    # Step 2 — embed
    model = "large" if plan.embed_model == "large" else "small"
    query_vector = await embed_query(body.question, model=model)

    if not query_vector:
        raise HTTPException(500, "Failed to embed query")

    # Step 3 — hybrid search + rerank (threshold derived internally from rerank scores)
    result: SearchResult = await hybrid_search(
        query_text=body.question,
        query_vector=query_vector,
        user_id=body.user_id,
        named_vector=plan.named_vector,
        alpha=plan.alpha,
        k=plan.k,
        score_threshold=0.0,
        rerank=True,
        rerank_top_n=min(plan.k, 15),
        rerank_threshold=0.3,   # drop only clearly irrelevant chunks
        collection_names=collection_names,
    )

    low_confidence = len(result.hits) == 0
    if low_confidence:
        return AskResponse(
            answer=_LOW_CONFIDENCE_MSG,
            references=[],
            user_id=body.user_id,
            collections_searched=collection_names,
            chunks_used=0,
            chunks_filtered=result.filtered_count,
            k_used=plan.k,
            alpha_used=plan.alpha,
            named_vector_used=plan.named_vector,
            tokens_used=0,
            router_reason=plan.reason,
            low_confidence=True,
            guardrail_passed=True,
            guardrail_verdict="SAFE",
            query_complexity=plan.complexity,
        )

    # Step 4 — generate answer
    answer_result: AnswerResult = await generate_answer(
        body.question,
        result.hits,
        k_used=plan.k,
        alpha_used=plan.alpha,
        named_vector_used=plan.named_vector,
    )

    return AskResponse(
        answer=answer_result.answer,
        references=[_ref_to_out(r) for r in answer_result.references],
        user_id=body.user_id,
        collections_searched=collection_names,
        chunks_used=len(result.hits),
        chunks_filtered=result.filtered_count,
        k_used=plan.k,
        alpha_used=plan.alpha,
        named_vector_used=plan.named_vector,
        tokens_used=answer_result.tokens_used,
        router_reason=plan.reason,
        low_confidence=False,
        guardrail_passed=answer_result.guardrail_passed,
        guardrail_verdict=answer_result.guardrail_verdict,
        query_complexity=plan.complexity,
    )
