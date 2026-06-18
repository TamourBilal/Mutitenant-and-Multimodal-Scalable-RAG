"""
Answer generator with inline [n] citations and a full references array.
Adapted from the original post_completion_rag.py RagllmText pattern.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI

from config import settings

logger = logging.getLogger(__name__)

_ANSWER_SYSTEM = """You are a precise, helpful assistant that answers questions based ONLY on the provided context chunks.

Rules:
1. Use ONLY information from the context. Do not hallucinate.
2. Cite every claim with inline markers like [1], [2], etc., matching the chunk numbers provided.
3. If the context does not contain enough information, say so explicitly.
4. Keep the answer concise and well-structured.
5. For legal documents, quote exact clause text when relevant.
6. If multiple chunks support a claim, cite all of them: [1][3].
"""

_ANSWER_USER_TEMPLATE = """Context chunks:
{context_block}

---
Question: {question}

Answer with inline citations ([1], [2], etc.):"""


@dataclass
class Reference:
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
    content_preview: str   # first 200 chars of the chunk


@dataclass
class AnswerResult:
    answer: str
    references: List[Reference] = field(default_factory=list)
    k_used: int = 0
    alpha_used: float = 0.0
    named_vector_used: str = ""
    tokens_used: int = 0


def _build_context_block(hits: List[Dict[str, Any]]) -> str:
    lines = []
    for i, hit in enumerate(hits, start=1):
        source_info = (
            f"[Source: {hit.get('filename', 'unknown')}"
            f", page {hit.get('page_no', 0)}"
            f", type={hit.get('chunk_type', 'text')}]"
        )
        lines.append(f"[{i}] {source_info}\n{hit.get('content', '')}")
    return "\n\n".join(lines)


def _hits_to_references(hits: List[Dict[str, Any]]) -> List[Reference]:
    refs = []
    for i, hit in enumerate(hits, start=1):
        refs.append(
            Reference(
                citation_index=i,
                filename=hit.get("filename", ""),
                source=hit.get("source", ""),
                doc_type=hit.get("doc_type", ""),
                page_no=int(hit.get("page_no") or 0),
                file_path=hit.get("file_path", ""),
                date=str(hit.get("date") or ""),
                score=float(hit.get("rerank_score") or hit.get("score") or 0.0),
                chunk_type=hit.get("chunk_type", "text"),
                doc_id=hit.get("doc_id", ""),
                content_preview=hit.get("content", "")[:200],
            )
        )
    return refs


_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(
            api_key=settings.OPENROUTER_API_KEY,
            base_url=settings.OPENROUTER_BASE_URL,
        )
    return _client


async def generate_answer(
    query: str,
    hits: List[Dict[str, Any]],
    *,
    k_used: int = 0,
    alpha_used: float = 0.0,
    named_vector_used: str = "",
) -> AnswerResult:
    """
    Generate a grounded answer with inline citations from retrieved chunks.
    """
    if not hits:
        return AnswerResult(
            answer="I could not find any relevant information in your documents to answer this question.",
            k_used=k_used,
            alpha_used=alpha_used,
            named_vector_used=named_vector_used,
        )

    context_block = _build_context_block(hits)
    user_message = _ANSWER_USER_TEMPLATE.format(
        context_block=context_block,
        question=query,
    )

    try:
        client = _get_client()
        response = await client.chat.completions.create(
            model=settings.ANSWER_MODEL,
            messages=[
                {"role": "system", "content": _ANSWER_SYSTEM},
                {"role": "user",   "content": user_message},
            ],
            max_tokens=1500,
            temperature=0.1,
        )
        answer_text = response.choices[0].message.content or ""
        tokens = response.usage.total_tokens if response.usage else 0

        logger.info(
            "[ANSWER_GEN] Done | query_len=%d hits=%d tokens=%d",
            len(query), len(hits), tokens,
        )
        return AnswerResult(
            answer=answer_text.strip(),
            references=_hits_to_references(hits),
            k_used=k_used,
            alpha_used=alpha_used,
            named_vector_used=named_vector_used,
            tokens_used=tokens,
        )

    except Exception as e:
        logger.error("[ANSWER_GEN] Failed | err=%s", e)
        return AnswerResult(
            answer=f"Error generating answer: {e}",
            references=_hits_to_references(hits),
            k_used=k_used,
            alpha_used=alpha_used,
            named_vector_used=named_vector_used,
        )
