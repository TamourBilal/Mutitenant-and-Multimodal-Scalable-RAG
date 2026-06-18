from __future__ import annotations

import uuid
from datetime import datetime
from typing import Annotated, Any, Dict, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from api.deps import get_current_user
from db.models import Document, User
from db.session import get_db
from intent.detector import DocumentIntent, detect_intent
from pipeline.ingest_pipeline import run_ingestion_pipeline

router = APIRouter()

MAX_FILE_SIZE_MB = 50


class IntentOut(BaseModel):
    doc_type: str
    chunking_strategy: str
    embed_model: str
    named_vector: str
    is_legal: bool
    is_table_heavy: bool
    is_scanned: bool
    confidence: float
    detection_method: str
    reason: str


class IngestResponse(BaseModel):
    doc_id: str
    filename: str
    source: str
    status: str
    message: str
    intent: IntentOut    # immediately available — from intent detection before background task


async def _background_ingest(
    file_bytes: bytes,
    filename: str,
    user_id: str,
    doc_id: str,
    doc_type_hint: Optional[str],
    db_url: str,
) -> None:
    """Runs in FastAPI BackgroundTasks — uses its own DB session."""
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

    engine = create_async_engine(db_url, echo=False)
    Session = async_sessionmaker(engine, expire_on_commit=False)

    result = await run_ingestion_pipeline(
        file_bytes=file_bytes,
        filename=filename,
        user_id=user_id,
        doc_id=doc_id,
        doc_type_hint=doc_type_hint,
    )

    async with Session() as session:
        from sqlalchemy import select
        row = await session.execute(select(Document).where(Document.id == doc_id))
        doc = row.scalar_one_or_none()
        if doc:
            doc.status         = result["status"]
            doc.chunk_count    = result["chunk_count"]
            doc.error_msg      = result.get("error")
            doc.doc_date       = result.get("doc_date")
            doc.doc_date_source = result.get("doc_date_source", "ingestion")
            doc.metadata_json  = result.get("metadata_json", "{}")
            await session.commit()

    await engine.dispose()


@router.post("", response_model=IngestResponse, status_code=202)
async def ingest_document(
    background_tasks: BackgroundTasks,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    file: UploadFile = File(...),
    doc_type: Optional[str] = Form(
        None,
        description="Override detected type: legal | pdf | html | image | other",
    ),
) -> IngestResponse:
    """
    Upload a document for ingestion.

    The **intent layer** runs immediately and returns the detected document
    classification in the response (doc_type, chunking strategy, embedding model,
    legal/table-heavy flags). Actual parsing and vector storage happen
    asynchronously. Poll `GET /api/v1/documents/{doc_id}` to check status.
    """
    file_bytes = await file.read()
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(400, f"File too large ({size_mb:.1f} MB). Limit: {MAX_FILE_SIZE_MB} MB.")
    if not file.filename:
        raise HTTPException(400, "Filename is required")

    # ── Intent detection (synchronous-ish, runs before background task) ────────
    intent: DocumentIntent = await detect_intent(
        file.filename, file_bytes, doc_type_hint=doc_type
    )

    doc_id = str(uuid.uuid4())
    doc = Document(
        id=doc_id,
        user_id=current_user.id,
        filename=file.filename,
        source=intent.source,
        doc_type=intent.doc_type,
        status="processing",
        chunk_count=0,
        created_at=datetime.utcnow(),
    )
    db.add(doc)
    await db.flush()

    from config import settings as cfg
    background_tasks.add_task(
        _background_ingest,
        file_bytes=file_bytes,
        filename=file.filename,
        user_id=current_user.id,
        doc_id=doc_id,
        doc_type_hint=doc_type,
        db_url=cfg.DB_URL,
    )

    return IngestResponse(
        doc_id=doc_id,
        filename=file.filename,
        source=intent.source,
        status="processing",
        message="Document queued. Poll GET /api/v1/documents/{doc_id} for status.",
        intent=IntentOut(
            doc_type=intent.doc_type,
            chunking_strategy=intent.chunking_strategy,
            embed_model=intent.embed_model,
            named_vector=intent.named_vector,
            is_legal=intent.is_legal,
            is_table_heavy=intent.is_table_heavy,
            is_scanned=intent.is_scanned,
            confidence=round(intent.confidence, 3),
            detection_method=intent.detection_method,
            reason=intent.reason,
        ),
    )
