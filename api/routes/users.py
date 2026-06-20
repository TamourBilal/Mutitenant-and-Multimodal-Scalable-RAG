from __future__ import annotations

import asyncio
import shutil
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated, List

from fastapi import APIRouter, Depends, HTTPException, status
from jose import jwt
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from db.models import Document, User
from db.session import get_db
from weaviate_store.client import delete_tenant, ensure_tenant

router = APIRouter()
pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto", bcrypt__rounds=12, truncate_error=False)


# ── Schemas ───────────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    email: EmailStr
    password: str


class AuthResponse(BaseModel):
    user_id: str
    email: str
    token: str


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class UserOut(BaseModel):
    user_id: str
    email: str
    created_at: str
    document_count: int


# ── Helpers ───────────────────────────────────────────────────────────────────

def _create_token(user_id: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    return jwt.encode(
        {"sub": user_id, "exp": expire},
        settings.SECRET_KEY,
        algorithm=settings.ALGORITHM,
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/register", response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
async def register(
    body: RegisterRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> AuthResponse:
    existing = await db.execute(select(User).where(User.email == body.email))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="Email already registered")

    user = User(
        id=str(uuid.uuid4()),
        email=body.email,
        password_hash=pwd_ctx.hash(body.password),
        created_at=datetime.utcnow(),
    )
    db.add(user)
    await db.flush()

    # Create a Weaviate tenant scoped to this user
    import asyncio
    await asyncio.to_thread(ensure_tenant, user.id)

    token = _create_token(user.id)
    return AuthResponse(user_id=user.id, email=user.email, token=token)


@router.get("", response_model=List[UserOut])
async def list_users(db: Annotated[AsyncSession, Depends(get_db)]) -> List[UserOut]:
    """
    List all effective users — registered accounts AND any tenant (user_id)
    that owns documents. The app uses user_id as a free-form tenant, so a tenant
    can own data without a Users row; those are reported with email '(tenant)'.
    """
    # Count only successfully-ingested documents (status='done'); processing /
    # error rows are not usable documents and would inflate the count.
    counts = dict(
        (await db.execute(
            select(Document.user_id, func.count(Document.id))
            .where(Document.status == "done")
            .group_by(Document.user_id)
        )).all()
    )
    # Also track total rows so tenant-only users with only failed docs still appear
    totals = dict(
        (await db.execute(
            select(Document.user_id, func.count(Document.id)).group_by(Document.user_id)
        )).all()
    )
    result = await db.execute(select(User).order_by(User.created_at.desc()))
    users = result.scalars().all()

    out: List[UserOut] = []
    seen = set()
    for u in users:
        seen.add(u.id)
        out.append(UserOut(
            user_id=u.id,
            email=u.email,
            created_at=u.created_at.isoformat() if u.created_at else "",
            document_count=int(counts.get(u.id, 0)),
        ))
    # Tenant-only users (have documents but no registered account)
    for tenant_id in totals:
        if tenant_id not in seen:
            out.append(UserOut(
                user_id=tenant_id,
                email="(tenant — not registered)",
                created_at="",
                document_count=int(counts.get(tenant_id, 0)),
            ))
    return out


@router.delete("/{user_id}", status_code=200)
async def delete_user(
    user_id: str,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> dict:
    """Delete a user/tenant and ALL their data: documents, vectors, tenant, and files."""
    from sqlalchemy import delete as sa_delete

    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()

    # Count documents so we can 404 only when nothing exists under this id
    doc_total = (await db.execute(
        select(func.count(Document.id)).where(Document.user_id == user_id)
    )).scalar() or 0
    if not user and doc_total == 0:
        raise HTTPException(404, "No user or documents found for this id")

    email = user.email if user else "(tenant — not registered)"

    # Remove the Weaviate tenant (drops all their vectors at once)
    await asyncio.to_thread(delete_tenant, user_id)

    # Remove local storage (originals, images, metadata)
    for base in (settings.STORAGE_DIR, settings.METADATA_DIR):
        user_dir = Path(base) / user_id
        if user_dir.exists():
            shutil.rmtree(user_dir, ignore_errors=True)

    # Remove Document rows, then the user row if it exists
    await db.execute(sa_delete(Document).where(Document.user_id == user_id))
    if user:
        await db.delete(user)

    return {"status": "deleted", "user_id": user_id, "email": email, "documents_removed": int(doc_total)}


@router.post("/login", response_model=AuthResponse)
async def login(
    body: LoginRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> AuthResponse:
    result = await db.execute(select(User).where(User.email == body.email))
    user = result.scalar_one_or_none()
    if not user or not pwd_ctx.verify(body.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = _create_token(user.id)
    return AuthResponse(user_id=user.id, email=user.email, token=token)
