from __future__ import annotations

import shutil
from pathlib import Path
from typing import Annotated, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.deps import get_current_user
from config import settings
from db.models import Document, User
from db.session import get_db
from weaviate_store.ingestor import delete_document_vectors

router = APIRouter()


class DocumentOut(BaseModel):
    doc_id: str
    filename: str
    source: str
    doc_type: str
    status: str
    chunk_count: int
    file_path: Optional[str]
    created_at: str
    error_msg: Optional[str]
    doc_date: Optional[str] = None           # publication/creation date extracted from content
    doc_date_source: Optional[str] = None    # "extracted" | "ingestion"
    is_legal: bool = False
    is_image: bool = False


@router.get("", response_model=List[DocumentOut])
async def list_documents(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    status_filter: Optional[str] = None,
    doc_type_filter: Optional[str] = None,
) -> List[DocumentOut]:
    """List all documents for the authenticated user."""
    stmt = select(Document).where(Document.user_id == current_user.id)
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
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> DocumentOut:
    result = await db.execute(
        select(Document).where(Document.id == doc_id, Document.user_id == current_user.id)
    )
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(404, "Document not found")

    return DocumentOut(
        doc_id=doc.id,
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
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> dict:
    """Delete a document: removes Weaviate vectors, SQLite row, and local files."""
    result = await db.execute(
        select(Document).where(Document.id == doc_id, Document.user_id == current_user.id)
    )
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(404, "Document not found")

    # Delete Weaviate vectors
    await delete_document_vectors(doc_id, current_user.id)

    # Delete local files
    doc_dir = Path(settings.STORAGE_DIR) / current_user.id / doc_id
    if doc_dir.exists():
        shutil.rmtree(doc_dir, ignore_errors=True)

    # Delete SQLite row
    await db.delete(doc)

    return {"status": "deleted", "doc_id": doc_id}
