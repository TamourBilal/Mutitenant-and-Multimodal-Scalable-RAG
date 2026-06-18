from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


def _uuid() -> str:
    return str(uuid.uuid4())


class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=_uuid)
    email = Column(String, unique=True, nullable=False, index=True)
    password_hash = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    documents = relationship("Document", back_populates="user", cascade="all, delete-orphan")


class Document(Base):
    __tablename__ = "documents"

    id = Column(String, primary_key=True, default=_uuid)
    user_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    filename = Column(String, nullable=False)
    source = Column(String, nullable=False)    # "pdf" | "html" | "image" | "other"
    doc_type = Column(String, nullable=False)  # "legal" | "pdf" | "html" | "image" | "other"
    status = Column(String, nullable=False, default="processing")  # processing | done | error
    chunk_count = Column(Integer, default=0)
    file_path = Column(Text, nullable=True)     # absolute local path to original file
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    error_msg = Column(Text, nullable=True)
    # Extended metadata
    doc_date = Column(DateTime, nullable=True)  # extracted from content; falls back to created_at
    doc_date_source = Column(String, nullable=True)   # "extracted" | "ingestion"
    metadata_json = Column(Text, nullable=True)        # JSON: {word_count, language, …}

    user = relationship("User", back_populates="documents")
