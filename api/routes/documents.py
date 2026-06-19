from __future__ import annotations

import shutil
from pathlib import Path
from typing import Annotated, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from db.models import Document
from db.session import get_db
from weaviate_store.ingestor import delete_document_vectors

router = APIRouter()


class DocumentOut(BaseModel):
    doc_id: str
    user_id: str
    filename: str
    source: str
    doc_type: str
    status: str
    chunk_count: int
    file_path: Optional[str]
    created_at: str
    error_msg: Optional[str]
    doc_date: Optional[str] = None
    doc_date_source: Optional[str] = None
    is_legal: bool = False
    is_image: bool = False


@router.get("", response_model=List[DocumentOut])
async def list_documents(
    user_id: str,
    db: Annotated[AsyncSession, Depends(get_db)],
    status_filter: Optional[str] = None,
    doc_type_filter: Optional[str] = None,
) -> List[DocumentOut]:
    """List all documents for a user. Pass `user_id` as a query param."""
    if not user_id.strip():
        raise HTTPException(400, "user_id is required")

    stmt = select(Document).where(Document.user_id == user_id)
    if status_filter:
        stmt = stmt.where(Document.status == status_filter)
    if doc_type_filter:
        stmt = stmt.where(Document.doc_type == doc_type_filter)
    stmt = stmt.order_by(Document.created_at.desc())

    result = await db.execute(stmt)
    docs = result.scalars().all()

    return [
        DocumentOut(
            doc_id=d.id,
            user_id=d.user_id,
            filename=d.filename,
            source=d.source,
            doc_type=d.doc_type,
            status=d.status,
            chunk_count=d.chunk_count or 0,
            file_path=d.file_path,
            created_at=d.created_at.isoformat() if d.created_at else "",
            error_msg=d.error_msg,
            doc_date=d.doc_date.isoformat() if d.doc_date else None,
            doc_date_source=d.doc_date_source,
            is_legal=d.doc_type == "legal",
            is_image=d.doc_type == "image",
        )
        for d in docs
    ]


@router.get("/{doc_id}", response_model=DocumentOut)
async def get_document(
    doc_id: str,
    user_id: str,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> DocumentOut:
    """Get a single document by ID. Pass `user_id` as a query param."""
    if not user_id.strip():
        raise HTTPException(400, "user_id is required")

    result = await db.execute(
        select(Document).where(Document.id == doc_id, Document.user_id == user_id)
    )
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(404, "Document not found")

    return DocumentOut(
        doc_id=doc.id,
        user_id=doc.user_id,
        filename=doc.filename,
        source=doc.source,
        doc_type=doc.doc_type,
        status=doc.status,
        chunk_count=doc.chunk_count or 0,
        file_path=doc.file_path,
        created_at=doc.created_at.isoformat() if doc.created_at else "",
        error_msg=doc.error_msg,
        doc_date=doc.doc_date.isoformat() if doc.doc_date else None,
        doc_date_source=doc.doc_date_source,
        is_legal=doc.doc_type == "legal",
        is_image=doc.doc_type == "image",
    )


@router.delete("/{doc_id}", status_code=200)
async def delete_document(
    doc_id: str,
    user_id: str,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> dict:
    """Delete a document and all its vectors. Pass `user_id` as a query param."""
    if not user_id.strip():
        raise HTTPException(400, "user_id is required")

    result = await db.execute(
        select(Document).where(Document.id == doc_id, Document.user_id == user_id)
    )
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(404, "Document not found")

    await delete_document_vectors(doc_id, user_id)

    doc_dir = Path(settings.STORAGE_DIR) / user_id / doc_id
    if doc_dir.exists():
        shutil.rmtree(doc_dir, ignore_errors=True)

    await db.delete(doc)

    return {"status": "deleted", "doc_id": doc_id, "user_id": user_id}
