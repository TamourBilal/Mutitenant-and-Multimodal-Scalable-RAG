"""
Agentic Multimodal RAG — FastAPI entry point.

Multi-tenant RAG with:
  - Weaviate (named vectors: text_small, text_large, image/CLIP)
  - OpenRouter (gpt-4o-mini for routing, gpt-4o for answers)
  - Adaptive query routing & cross-encoder reranking
  - Async document ingestion with intent detection
  - JWT multi-user auth with per-user Weaviate tenant isolation
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routes import ask, documents, ingest, query, search
from config import settings
from db.session import init_db
from weaviate_store.client import close_weaviate_client, get_weaviate_client
from weaviate_store.schema import ensure_schema

logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=== RAG System Starting ===")

    await init_db()
    logger.info("SQLite schema initialized")

    client = get_weaviate_client()
    ensure_schema(client)
    logger.info("Weaviate schema ensured")

    yield

    logger.info("=== RAG System Shutting Down ===")
    close_weaviate_client()





app = FastAPI(
    title=settings.APP_NAME,
    description=(
        "Multi-tenant multimodal RAG: Weaviate + OpenRouter + agentic query routing.\n\n"
        "**Auth**: Register → receive JWT → pass as `Authorization: Bearer <token>`."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ────────────────────────────────────────────────────────────────────
app.include_router(ingest.router,    prefix="/api/v1/ingest",    tags=["ingestion"])
app.include_router(documents.router, prefix="/api/v1/documents", tags=["documents"])
app.include_router(search.router,    prefix="/api/v1/search",    tags=["retrieval"])
app.include_router(ask.router,       prefix="/api/v1/ask",       tags=["generation"])
app.include_router(query.router,     prefix="/api/v1/query",     tags=["generation"])


# ── Utility endpoints ─────────────────────────────────────────────────────────

@app.get("/health", tags=["utility"])
async def health_check() -> dict:
    """Weaviate liveness probe."""
    try:
        client = get_weaviate_client()
        ready = client.is_ready()
        return {"status": "healthy" if ready else "degraded", "weaviate": ready}
    except Exception as exc:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "detail": str(exc)},
        )


@app.get("/api/v1/routes", tags=["utility"])
async def list_routes() -> dict:
    """Enumerate all registered API paths."""
    return {
        "routes": [
            {"path": r.path, "methods": sorted(r.methods), "name": r.name}
            for r in app.routes
            if hasattr(r, "methods") and hasattr(r, "path")
        ]
    }


# ── Exception handlers ────────────────────────────────────────────────────────

@app.exception_handler(HTTPException)
async def http_exc_handler(request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.exception_handler(Exception)
async def generic_exc_handler(request, exc: Exception):
    logger.error("Unhandled error: %s", exc, exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        timeout_graceful_shutdown=3,  # force-kill background tasks after 3s on CTRL+C
    )
