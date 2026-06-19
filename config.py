from __future__ import annotations

from typing import List

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── FastAPI & Server ──────────────────────────────────────────────────────
    APP_NAME: str = "Agentic Multimodal RAG"
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # ── Database (SQLite async) ───────────────────────────────────────────────
    DB_URL: str = "sqlite+aiosqlite:///./rag.db"

    # ── Weaviate Vector DB ────────────────────────────────────────────────────
    WEAVIATE_HOST: str = "localhost"
    WEAVIATE_PORT: int = 8080
    WEAVIATE_GRPC_PORT: int = 50051
    WEAVIATE_COLLECTION_DOCUMENTS: str = "RAGDocuments"
    WEAVIATE_COLLECTION_TABLES: str = "RAGTables"
    WEAVIATE_COLLECTION_IMAGES: str = "RAGImages"
    WEAVIATE_BATCH_SIZE: int = 500

    # ── OpenRouter (LLM + Embeddings) ─────────────────────────────────────────
    OPENROUTER_API_KEY: str
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    ROUTER_MODEL: str = "openai/gpt-4o-mini"
    ANSWER_MODEL: str = "openai/gpt-4o"

    # ── Embedding Models ──────────────────────────────────────────────────────
    EMBED_MODEL_SMALL: str = "text-embedding-3-small"
    EMBED_MODEL_LARGE: str = "text-embedding-3-large"
    EMBED_DIM_SMALL: int = 1536
    EMBED_DIM_LARGE: int = 3072
    EMBED_BATCH_SIZE: int = 100

    # ── Vision model for image captioning (OpenRouter) ───────────────────────
    VISION_MODEL: str = "openai/gpt-4o-mini"

    # ── Reranking model (OpenRouter) ──────────────────────────────────────────
    RERANK_MODEL: str = "openai/gpt-4o-mini"

    # ── Chunking ──────────────────────────────────────────────────────────────
    SEMANTIC_BREAKPOINT_PERCENTILE: int = 95
    HTML_CHUNK_SIZE: int = 1000
    HTML_CHUNK_OVERLAP: int = 100

    # ── Retrieval Defaults ────────────────────────────────────────────────────
    SCORE_THRESHOLD: float = 0.70
    DEFAULT_K: int = 10
    DEFAULT_ALPHA: float = 0.75   # 75% semantic, 25% BM25

    # ── JWT ───────────────────────────────────────────────────────────────────
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    # ── File Storage ──────────────────────────────────────────────────────────
    STORAGE_DIR: str = "./storage"
    METADATA_DIR: str = "./storage/metadata"
    MAX_FILE_SIZE_MB: int = 50

    # ── Intent Detection Keyword Lists ───────────────────────────────────────
    LEGAL_KEYWORDS: List[str] = [
        "contract", "agreement", "terms", "conditions", "liability", "indemnif",
        "warrant", "clause", "statute", "regulation", "compliance", "policy",
        "nda", "confidential", "arbitrat", "jurisdiction", "govern",
    ]
    NEWS_KEYWORDS: List[str] = [
        "news", "article", "press", "release", "breaking", "headline",
        "report", "correspondent", "journalist", "publication", "byline",
    ]
    MEDICAL_KEYWORDS: List[str] = [
        "patient", "clinical", "diagnosis", "treatment", "medical",
        "hospital", "doctor", "drug", "health", "disease", "symptom",
    ]
    FINANCIAL_KEYWORDS: List[str] = [
        "revenue", "earnings", "financial", "balance", "sec", "filing",
        "stock", "investor", "quarterly", "annual", "statement",
    ]
    RESEARCH_KEYWORDS: List[str] = [
        "research", "study", "abstract", "paper", "academic",
        "scientific", "hypothesis", "methodology", "conclusion", "pubmed",
    ]

    model_config = {"env_file": ".env", "case_sensitive": True}


settings = Settings()
