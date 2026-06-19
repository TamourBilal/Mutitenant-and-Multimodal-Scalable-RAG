# RAG System — Design Decisions & Technical Rationale

**Author:** Tamour Bilal
**System:** Agentic RAG — Multi-tenant, Multi-modal, Weaviate + OpenRouter
**Stack:** FastAPI · Weaviate v4 · OpenRouter · SQLite · PyMuPDF · gpt-4o-mini (routing/reranking/guardrail/captioning) · gpt-4o (answers)

---

## Current Architecture (as-built)

```
User Request (user_id + collections + query)
         │
         ▼
┌─────────────────────────────────────────────────────┐
│                    FastAPI (main.py)                 │
│                                                      │
│  POST /ingest          POST /ask    POST /search     │
│  POST /query           GET /docs    DELETE /docs     │
└──────────────┬──────────────────────────────────────┘
               │
    ┌──────────▼──────────┐
    │   Guardrail Check   │  gpt-4o-mini → SAFE / UNSAFE
    └──────────┬──────────┘
               │ SAFE
    ┌──────────▼──────────┐
    │    Query Router     │  gpt-4o-mini → k(5-40), alpha,
    │                     │  named_vector, complexity
    └──────────┬──────────┘
               │
    ┌──────────▼──────────┐
    │   Embed Query       │  text-embedding-3-small (1536d)
    │                     │  text-embedding-3-large (3072d, legal only)
    └──────────┬──────────┘
               │
    ┌──────────▼──────────────────────────────────────┐
    │         Weaviate Hybrid Search                   │
    │  BM25 (keyword) + Semantic (vector), alpha=0-1   │
    │  Per-user tenant isolation                       │
    │  Collections: RAGDocuments · RAGTables · RAGImages│
    └──────────┬──────────────────────────────────────┘
               │
    ┌──────────▼──────────┐
    │    Deduplication    │  by (page_no, content[:80])
    └──────────┬──────────┘
               │
    ┌──────────▼──────────┐
    │   LLM Reranker      │  gpt-4o-mini → scores 0.0-1.0
    │                     │  fallback: hybrid score
    └──────────┬──────────┘
               │ score ≥ 0.3 passes
    ┌──────────▼──────────┐
    │   Answer Generator  │  gpt-4o → markdown + tables
    │                     │  inline [n] citations
    │                     │  📄 file — page N references
    └─────────────────────┘
```

---

## 1. Chunking Strategy

### Why content-type-aware chunking?

A single strategy fails across modalities. A legal contract and a news article
have nothing in common structurally. Three paths, driven by **intent detection**:

### Path A — PDF documents (text + tables)

**Approach:** Custom `split_semantic` in `chunking/semantic_chunker.py`.

- Splits text into sentences with regex
- Embeds **all sentences in one batched API call** (not per-pair like LangChain SemanticChunker)
- Detects boundaries where cosine similarity drops below the `SEMANTIC_BREAKPOINT_PERCENTILE`
- Groups sentences into chunks at those boundaries

**Why custom instead of LangChain SemanticChunker?**
LangChain's SemanticChunker made N×(N-1) API calls — one per sentence pair. For a 66-page PDF
this caused hundreds of sequential HTTP calls, blocking the event loop for minutes. The custom
implementation makes **one batched API call** for all sentences, reducing ingestion time from
minutes to seconds.

**Parser:** PyMuPDF (fitz) replaced pdfplumber. pdfplumber blocked indefinitely on some PDFs;
PyMuPDF opens the entire file in memory and processes all pages in one pass.

**Demo mode:** First 20 pages only (`max_pages=20` in `parsing/pdf_parser.py`).

### Path B — HTML pages

**Approach:** `split_html` in `chunking/recursive_html.py` — custom recursive splitter
with no external dependencies. Separators: `["\n\n", "\n", ". ", " ", ""]`,
chunk size `HTML_CHUNK_SIZE=1000`, overlap `HTML_CHUNK_OVERLAP=100`.

**Why no LangChain?** LangChain was removed to eliminate the `langchain_text_splitters`
dependency conflict. The custom splitter is ~50 lines and covers all cases.

### Path C — Images

**Approach:** OpenRouter `VISION_MODEL` (default: `gpt-4o-mini`) generates a
2-5 sentence caption. Caption embedded with `text-embedding-3-small` → stored
in `RAGImages` as `text_small` (1536-dim).

**Why no CLIP?** CLIP (sentence-transformers, HuggingFace) was removed. The 512-dim
CLIP space is disjoint from the 1536-dim text space used for everything else, making
cross-collection search inconsistent. Caption-based text embeddings keep all content
in the **same vector space** — a text query about "bar chart" retrieves image captions
using the same pipeline as document search.

---

## 2. Embedding Models

| Content type | Model | Dimensions | Rationale |
|---|---|---|---|
| PDF, HTML, images | `text-embedding-3-small` | 1536 | Cost-effective, consistent space across all content types |
| Legal/compliance PDFs | `text-embedding-3-large` | 3072 | Denser space for domain-specific clause distinctions |
| Query (non-legal) | `text-embedding-3-small` | 1536 | Matches ingestion space |
| Query (legal) | `text-embedding-3-large` | 3072 | Matches ingestion space |

**Single API provider:** All embeddings go through OpenRouter. No local models needed.
This simplifies deployment — no GPU, no HuggingFace downloads, no sentence-transformers.

**Named vector override rule:** Routes force `text_small` unless `doc_type_filter=legal`.
This prevents the router from suggesting `text_large` for non-legal queries that would
search an empty vector space.

---

## 3. Weaviate Architecture — Multi-Tenancy & Collections

### Three-collection design

```
Weaviate cluster
│
├── RAGDocuments   (text_small: 1536d, text_large: 3072d)
│   └── Tenant: alice   ← isolated HNSW index
│   └── Tenant: bob
│
├── RAGTables      (text_small: 1536d, text_large: 3072d)
│   └── Tenant: alice
│
└── RAGImages      (text_small: 1536d)
    └── Tenant: alice
```

### Collection aliases (user-facing)

| User passes | Internal collection | Auto doc_type filter |
|-------------|--------------------|--------------------|
| `documents` | `RAGDocuments` | none |
| `html` | `RAGDocuments` | `doc_type=html` |
| `tables` | `RAGTables` | none |
| `images` | `RAGImages` | none |

Both `documents` and `RAGDocuments` are accepted (and any combination).

### Tenant lifecycle

Tenants are auto-created at ingest via `ensure_tenant(user_id)`. No registration
endpoint required — any `user_id` string becomes a valid isolated tenant on first upload.

### Batch upsert

500 objects per batch via gRPC. `collection_name_override` allows the user-selected
collection to override the default chunk_type-based routing.

---

## 4. Agentic Query Router

The router (`agent/query_router.py`) calls `gpt-4o-mini` to produce a `QueryPlan`:

| Field | Purpose |
|-------|---------|
| `complexity` | `simple`/`moderate`/`complex` → determines k bounds |
| `k` | Clamped: simple=5-10, moderate=10-20, complex=25-40 |
| `alpha` | 0.3=BM25-heavy, 0.75=semantic-heavy, 0.5=balanced |
| `named_vector` | `text_small` (default) or `text_large` (legal only) |
| `embed_model` | `small` or `large` |
| `score_threshold` | Per-complexity defaults: simple=0.75, moderate=0.65, complex=0.55 |

**Key rule:** `text_large` is forced back to `text_small` for any non-legal query,
regardless of what the router suggests. This prevents searching an empty vector space
when documents were indexed with `text_small`.

---

## 5. Reranker

**Replaced:** `cross-encoder/ms-marco-MiniLM-L-6-v2` (local HuggingFace)
**Current:** `gpt-4o-mini` via OpenRouter

**Why switched?**
- Removes `sentence-transformers` (~500MB) dependency
- No startup download / warm-up delay
- LLM reranker understands query intent better than a classification cross-encoder
- Fallback: if LLM fails or returns partial scores, **hybrid scores are pre-assigned**
  so no chunk ever silently receives `rerank_score=0.0`

**Score mapping:**
- Chunks indexed 1-based `[1]...[n]` in prompt
- Score map tried with both 1-based and 0-based lookup for robustness
- Missing scores → hybrid score as fallback (logged as warning)

**Rerank threshold:** `0.3` (ask endpoint). Chunks below 0.3 are dropped before
reaching the LLM. Search endpoint has no threshold — returns all ranked results.

---

## 6. Guardrail

A fast pre-flight check using `gpt-4o-mini` before any retrieval:

```
Query → gpt-4o-mini (SAFE/UNSAFE) → blocked if UNSAFE
```

UNSAFE triggers: prompt injection, harmful content requests, off-topic queries.
Returns `guardrail_passed=false` with a fixed message — no LLM call for answers.

---

## 7. API Design

### Auth — No JWT (simplified)

JWT was removed. `user_id` is passed explicitly in every request body or query param.
This trades security for simplicity — appropriate for a single-developer RAG backend
where the caller controls user identity. Re-adding JWT is a one-file change in `api/deps.py`.

### Endpoints

| Method | Path | Auth | Purpose |
|--------|------|------|---------|
| `POST` | `/api/v1/ingest` | user_id (form) | Upload file → async ingestion (202) |
| `GET` | `/api/v1/documents` | user_id (query) | List documents |
| `GET` | `/api/v1/documents/{doc_id}` | user_id (query) | Poll ingestion status |
| `DELETE` | `/api/v1/documents/{doc_id}` | user_id (query) | Delete doc + vectors + files |
| `POST` | `/api/v1/search` | user_id (body) | Hybrid search, raw chunks, no LLM |
| `POST` | `/api/v1/ask` | user_id (body) | Full RAG: search → rerank → answer |
| `POST` | `/api/v1/query` | user_id (body) | Alt RAG endpoint |
| `GET` | `/health` | none | Weaviate liveness |
| `GET` | `/api/v1/routes` | none | List endpoints |

### Request shapes

**`/ask`**
```json
{
  "question": "Compare GDP across African regions with a table",
  "user_id": "alice",
  "collections": ["documents"],
  "k": 10,
  "alpha": 0.5
}
```

**`/search`**
```json
{
  "query": "GDP statistics by region",
  "user_id": "alice",
  "collections": ["documents", "tables"],
  "k": 10,
  "alpha": 0.5
}
```

**`/ingest`** (form-data)
```
file        = document.pdf
user_id     = alice
collection  = documents | tables | images | html
doc_type    = pdf | legal | html | image | financial | medical | research (optional)
```

---

## 8. Removed Dependencies

| Removed | Replaced by | Reason |
|---------|-------------|--------|
| `sentence-transformers` (CLIP) | OpenRouter text embeddings | Eliminates local model, keeps all vectors in same space |
| `sentence-transformers` (cross-encoder) | gpt-4o-mini reranker | Eliminates 500MB download, better intent understanding |
| `langchain` / `langchain-openai` / `langchain-experimental` | Custom `split_semantic` | LangChain SemanticChunker made N² API calls; custom impl uses 1 batch call |
| `langchain_text_splitters` | Custom `split_html` | Dependency conflict; custom is simpler |
| `pdfplumber` | PyMuPDF (fitz) | pdfplumber blocked indefinitely on some PDFs |
| JWT auth (`python-jose`) | Explicit `user_id` param | Simplified auth for single-developer use |

---

## 9. Scale & Cost Estimates

### 100K documents → ~500K vectors

| Phase | Approach | Time estimate |
|---|---|---|
| Parsing (PyMuPDF) | Async background tasks | ~4 hrs (100 docs/hr per worker) |
| Embedding (small, 400K chunks) | OpenRouter batch=100 | ~4 hrs |
| Embedding (large, 100K legal) | OpenRouter batch=100 | ~2 hrs |
| Weaviate upsert (500K, 500/batch) | 1000 gRPC batch calls | ~30 min |

### Monthly cost at steady state (100 active users, 100K queries/mo)

| Item | Volume | Unit cost | Monthly |
|---|---|---|---|
| Embedding (new docs) | 500K chunks/mo | $0.02/1M tokens | ~$5 |
| Routing + reranking (gpt-4o-mini) | 100K queries × 3 calls | $0.15/1M tokens | ~$6 |
| Answers (gpt-4o) | 100K queries × 2K tokens | $2.50/1M tokens | ~$500 |
| Weaviate cloud (5M vectors) | 5M vectors | $0.05/1M/mo | ~$250 |
| **Total** | | | **~$761/mo** |

**Optimisation:** Cache common answers (Redis, 1hr TTL, keyed by embedding similarity)
reduces LLM answer spend ~40%. Image captioning cost (gpt-4o-mini vision) is negligible
unless the corpus is image-heavy.

**What breaks first at 1M documents:**
1. Single Weaviate node → HNSW index exceeds RAM → enable PQ compression or shard
2. SQLite → migrate to PostgreSQL (one connection string change, SQLAlchemy abstracts it)
3. OpenRouter rate limits → add exponential backoff + async queue (ARQ or Celery)
