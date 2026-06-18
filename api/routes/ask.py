"""
POST /api/v1/ask

Full RAG endpoint — hybrid search (with score threshold) → cross-encoder
rerank → LLM answer generation with inline [n] citations.

Returns a generated answer only when the retrieved context clears the
relevance threshold; otherwise responds with a low-confidence fallback
message rather than hallucinating.
"""
from __future__ import annotations

from typing import Annotated, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from agent.answer_gen import AnswerResult, Reference, generate_answer
from agent.query_router import route_query
from api.deps import get_current_user
from config import settings
from db.models import User
from db.session import get_db
from embedding.image_embedder import embed_text_clip
from embedding.text_embedder import embed_query
from weaviate_store.retriever import SearchResult, hybrid_search

router = APIRouter()

_LOW_CONFIDENCE_MSG = (
    "I could not find any document chunks that met the relevance threshold "
    "({threshold:.0%}) for your question. Try rephrasing, lowering the threshold, "
    "or ingesting more relevant documents."
)


# ── Request / Response schemas ────────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str
    doc_type_filter: Optional[str] = Field(
        None,
        description="Restrict to a doc_type: legal | news | medical | financial | research | pdf | html | image",
    )
    k: Optional[int] = Field(None, ge=1, le=50, description="Max context chunks (overrides dynamic k)")
    score_threshold: Optional[float] = Field(
        None, ge=0.0, le=1.0,
        description="Minimum hybrid score for a chunk to reach the LLM (default from config)",
    )
    alpha: Optional[float] = Field(None, ge=0.0, le=1.0, description="BM25/semantic blend")


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


class AskResponse(BaseModel):
    answer: str
    references: List[ReferenceOut]
    chunks_used: int              # chunks that passed threshold and reached the LLM
    chunks_filtered: int          # chunks dropped below threshold
    score_threshold: float
    k_used: int
    alpha_used: float
    named_vector_used: str
    tokens_used: int
    router_reason: str
    low_confidence: bool          # True when 0 chunks passed the threshold


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

@router.post("", response_model=AskResponse)
async def ask_question(
    body: AskRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    _db: Annotated[AsyncSession, Depends(get_db)],
) -> AskResponse:
    """
    Full RAG pipeline: retrieve → threshold filter → rerank → LLM answer.

    The LLM only receives chunks that cleared the **score_threshold**.
    If no chunks pass, the endpoint returns a `low_confidence=true` response
    with a fallback message instead of hallucinating an answer.

    Top-k and threshold are determined dynamically by the query router
    (GPT-4o-mini) and can be overridden per-request.
    """
    if not body.question.strip():
        raise HTTPException(400, "Question cannot be empty")

    # ── Step 1: Route ─────────────────────────────────────────────────────────
    plan = await route_query(
        body.question,
        doc_type_hint=body.doc_type_filter,
        k_override=body.k,
        threshold_override=body.score_threshold,
    )
    if body.alpha is not None:
        plan.alpha = max(0.0, min(1.0, body.alpha))

    # ── Step 2: Embed query ───────────────────────────────────────────────────
    if plan.embed_model == "clip":
        vecs = await embed_text_clip([body.question])
        query_vector = vecs[0] if vecs else []
    elif plan.embed_model == "large":
        query_vector = await embed_query(body.question, model="large")
    else:
        query_vector = await embed_query(body.question, model="small")

    if not query_vector:
        raise HTTPException(500, "Failed to embed query")

    # ── Step 3: Hybrid search + threshold gate ────────────────────────────────
    result: SearchResult = await hybrid_search(
        query_text=body.question,
        query_vector=query_vector,
        user_id=current_user.id,
        named_vector=plan.named_vector,
        alpha=plan.alpha,
        k=plan.k,
        doc_type_filter=plan.doc_type_filter,
        score_threshold=plan.score_threshold,
        rerank=True,
        rerank_top_n=min(plan.k, 8),
        collection_names=plan.collections,
    )

    # ── Step 4: Low-confidence gate ───────────────────────────────────────────
    low_confidence = len(result.hits) == 0
    if low_confidence:
        return AskResponse(
            answer=_LOW_CONFIDENCE_MSG.format(threshold=result.score_threshold),
            references=[],
            chunks_used=0,
            chunks_filtered=result.filtered_count,
            score_threshold=result.score_threshold,
            k_used=plan.k,
            alpha_used=plan.alpha,
            named_vector_used=plan.named_vector,
            tokens_used=0,
            router_reason=plan.reason,
            low_confidence=True,
        )

    # ── Step 5: Generate answer ───────────────────────────────────────────────
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
        chunks_used=len(result.hits),
        chunks_filtered=result.filtered_count,
        score_threshold=result.score_threshold,
        k_used=plan.k,
        alpha_used=plan.alpha,
        named_vector_used=plan.named_vector,
        tokens_used=answer_result.tokens_used,
        router_reason=plan.reason,
        low_confidence=False,
    )
