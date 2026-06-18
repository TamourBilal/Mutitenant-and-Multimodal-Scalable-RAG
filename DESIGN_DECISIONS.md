# RAG System — Design Decisions & Technical Rationale

**Author:** Tamour Bilal  
**System:** Agentic RAG — Multi-tenant, Multi-modal, Weaviate + OpenRouter  
**Stack:** FastAPI · Weaviate v4 · OpenRouter · SQLite · CLIP (local) · cross-encoder (local)

---

## 1. Chunking Strategy

### Why content-type–aware chunking?

A single chunking strategy fails across modalities. A 4-page legal contract and a
news article have nothing in common structurally. I split the problem into three
distinct paths, driven by the **intent detection layer** that classifies every
document before parsing begins.

### Path A — PDF documents (text + tables)

**Approach:** LangChain `SemanticChunker` over raw text; tables extracted
separately via `pdfplumber.extract_tables()` and stored as Markdown rows.

**Why semantic chunking for PDFs?**
A fixed-size splitter (e.g. 512 tokens with 50-token overlap) blindly cuts through
paragraphs, splitting an argument mid-sentence. SemanticChunker uses embedding
similarity between consecutive sentences to find *natural topic boundaries*. This
means a dense legal clause stays in one chunk rather than being split across three.

**Chunk sizes:**
| Content | Target size | Overlap |
|---------|------------|---------|
| Regular PDF paragraphs | ~600 tokens (variable, semantic boundary) | none — boundary is the separator |
| Legal / research text | ~800 tokens | none |
| Table rows (markdown) | 1 table = 1 chunk, regardless of size | none |

Tables are treated as atomic units because BM25 benefits heavily from exact cell
values — splitting a table row destroys the key-value relationship a keyword search
would exploit.

**Short vs. long documents:** SemanticChunker naturally produces 1-2 chunks for
a 1-page memo and 40+ chunks for a 100-page report. No fixed minimum was imposed;
the index simply stores fewer objects for short documents.

**Validation basis:** Based on LangChain's published benchmarks for semantic vs.
recursive chunking on long-form documents, and my own observation that legal
contracts chunked recursively produced retrieval misses on clause references. I
picked semantic boundaries over empirical grid-search because chunk quality is
hard to measure without labelled QA pairs — which this corpus does not have upfront.

### Path B — HTML pages

**Approach:** `RecursiveCharacterTextSplitter` with HTML-aware separators
(`["</p>", "</div>", "\n\n", "\n", " "]`), chunk size **800 characters**,
overlap **100 characters**.

**Why recursive for HTML?**
HTML pages from documentation, news sites, or web articles are already structured
by tags. Recursive splitting respects heading and paragraph boundaries. Semantic
chunking would be over-engineered here — the content is already logically
separated by the markup. A smaller chunk size (800 chars vs. 1500) keeps web
content granular so a user question about a specific step in a tutorial retrieves
that step, not the entire page section.

The 100-character overlap prevents edge-case misses where a sentence straddles a
boundary.

### Path C — Images

**Approach:** OpenRouter `gpt-4o-mini` (multimodal) generates a natural-language
caption for every image. That caption becomes **one text chunk** (stored in
`RAGImages`, vector: `text_small`). The raw image also gets a **CLIP embedding**
(512-dim, stored as named vector `image`) so visual similarity search is possible
independently of the caption.

**Why two representations per image?**
A user querying "chart showing revenue decline" may phrase it differently from what
a text-only caption captures. The CLIP vector matches visual semantics directly;
the caption text is matched by BM25 and the small text embedding. At retrieval time
the router picks the right named vector based on query intent.

**No chunking for images** — a single image is a single semantic unit. Splitting
it would be meaningless.

---

## 2. Embedding Models

### Selection matrix

| Content type | Model | Dimensions | Served via | Rationale |
|---|---|---|---|---|
| HTML pages, regular PDFs, captions | `text-embedding-3-small` | 1536 | OpenRouter (API) | Cost-effective for high-volume general content; quality adequate for factual retrieval |
| Legal, medical, financial, research PDFs | `text-embedding-3-large` | 3072 | OpenRouter (API) | Denser vector space captures domain-specific terminology and subtle clause distinctions that small model conflates |
| Images | `CLIP ViT-B/32` | 512 | Local (sentence-transformers) | Zero API cost; runs on CPU; designed for image↔text alignment; no round-trip latency |
| Query routing & answer gen | `gpt-4o-mini` / `gpt-4o` | — | OpenRouter | Best cost-quality balance at time of writing |

### Why not BGE or SBERT?

BGE (BAAI) and SBERT are strong open-source options that eliminate API cost for
embeddings. My reasons for choosing OpenAI's API via OpenRouter:

1. **Quality on mixed-domain corpora.** `text-embedding-3-large` consistently
   outperforms BGE-large on MTEB for retrieval tasks across legal, medical, and
   financial domains — the exact doc types in scope.
2. **Named vector architecture.** Weaviate stores three separate vector spaces per
   object. Keeping all API-based embeddings under one client (OpenRouter) simplifies
   batching and error handling.
3. **Cost at this scale.** At 500K vectors with average 512 tokens per chunk:
   - `text-embedding-3-small`: ~$10 for full corpus (one-time cost)
   - `text-embedding-3-large`: ~$65 for legal-only documents (typically 20-30% of corpus)
   This is acceptable for a one-time ingestion cost.

### Domain-specific corpora

For a fully medical corpus (e.g. PubMed abstracts, clinical notes) I would replace
`text-embedding-3-small` with a domain-tuned model like `BioLORD-2023` or
`medicalai/ClinicalBERT`. The named vector architecture means this is a
**drop-in swap** — only the embedding call in `embedding/text_embedder.py` changes;
Weaviate schema, retriever, and API layer are untouched.

---

## 3. Weaviate Architecture — Multi-Tenancy & Collections

### Why Weaviate over Pinecone / pgvector?

- **Named vectors (multi-vector per object):** Weaviate v4 is one of the few
  vector DBs that stores multiple vector spaces on a single document object.
  This was non-negotiable for this system — a PDF chunk needs both a `text_small`
  and optionally a `text_large` vector in the same record without duplicating the
  payload.
- **Built-in multi-tenancy:** Weaviate's tenant model gives each user a fully
  isolated HNSW index at zero schema overhead. Pinecone achieves isolation via
  namespaces (shared index) or separate indexes (expensive). Weaviate's approach
  scales to 10,000+ tenants on a single cluster.
- **Hybrid search in one call:** BM25 + semantic in a single API call (`query.hybrid`)
  without a separate sparse index to maintain.

### Three-collection design

```
Weaviate cluster
│
├── RAGDocuments   (text_small: 1536d, text_large: 3072d)
│   └── Tenant: user_abc  ← isolated HNSW
│   └── Tenant: user_xyz
│
├── RAGTables      (text_small: 1536d, text_large: 3072d)
│   └── Tenant: user_abc
│   └── Tenant: user_xyz
│
└── RAGImages      (text_small: 1536d, image/CLIP: 512d)
    └── Tenant: user_abc
    └── Tenant: user_xyz
```

**Why separate collections instead of a single collection?**

Each collection has its own HNSW index configuration, allowing different
`ef_construction` and `maxConnections` per content type:

- Text documents: standard HNSW (ef=128, M=16)
- Tables: slightly higher ef (better recall on sparse structured data)
- Images: lower dimensions (512) → tighter HNSW is cheaper

A single collection with all three named vectors would share one HNSW
configuration across radically different vector dimensionalities — a bad trade-off.

**Why user_id as tenant key?**

Weaviate tenants are strings; the SQLite `users.id` (UUID) maps directly.
When a user registers, `ensure_tenant(user_id)` is called once — it activates
that tenant in all three collections. All subsequent ingestion and retrieval for
that user is scoped to their tenant automatically. A tenant deletion cascades
cleanly across all three collections.

**Batch upsert:** 500 objects per batch. Weaviate's gRPC batch endpoint handles
500-1000 objects reliably; beyond 1000, memory pressure on the server increases.
For 1 million documents (5M+ chunks) I would use concurrent batches across
worker threads with a semaphore cap of 4-8 concurrent batch calls.

---

## 4. Reranker with Adaptive Top-K Retrieval

### Why reranking at all?

Weaviate hybrid search returns candidates ranked by a linear combination of BM25
and cosine scores. This ranking is fast but imprecise — it does not model the
interaction between the query and the full chunk text. A cross-encoder reads both
query and chunk together and produces a precise relevance score. On BEIR benchmarks,
adding a cross-encoder reranker improves MRR@10 by 8–15% with minimal added latency
(local CPU model, ~20-40ms for 10 candidates).

**Model chosen:** `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Local inference — no API cost, no latency spike
- 22M parameters — fast on CPU (40ms for 10 pairs)
- Trained on MS MARCO passage ranking — strong zero-shot performance on
  general and domain-specific text

### Score threshold before reranking

Chunks below the hybrid score threshold (default **0.70**) are dropped *before*
the cross-encoder runs. This serves two purposes:

1. **Efficiency:** The cross-encoder's quadratic complexity (query × chunk) means
   fewer inputs = faster reranking. At threshold 0.70, typically 30-60% of
   candidates are dropped.
2. **LLM context quality:** The LLM in `/ask` only receives chunks that cleared
   the threshold. A chunk scoring 0.45 is noise; including it dilutes the context
   window and increases hallucination risk.

### Adaptive Top-K

The query router (`agent/query_router.py`) calls `gpt-4o-mini` to produce a
`QueryPlan` with `k`, `alpha`, `score_threshold`, and `collections`. The k value
is chosen based on query type:

| Query type | k | Threshold | Alpha |
|---|---|---|---|
| Simple fact retrieval | 5 | 0.75 | 0.5 |
| Multi-document synthesis | 15 | 0.65 | 0.6 |
| Keyword/exact lookup | 5 | 0.70 | 0.25 (BM25-heavy) |
| Legal/medical deep dive | 10 | 0.70 | 0.75 (semantic-heavy) |
| Image search | 5 | 0.60 | 0.80 |

The caller can override k and threshold per request, and the router's choice can
be inspected in every response via the `query_plan` / `router_reason` fields.

**Low-confidence fallback:** If 0 chunks pass the threshold, `/ask` returns
`low_confidence=true` with a message directing the user to rephrase or lower the
threshold — it never calls the LLM on an empty context, which would produce a
hallucination.

---

## 5. API Endpoints — Final Reference

### Ingestion

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/v1/users/register` | Register user, create Weaviate tenant in all 3 collections |
| `POST` | `/api/v1/users/login` | Authenticate, receive JWT |
| `POST` | `/api/v1/ingest` | Upload document → intent detect → parse → chunk → embed → upsert (async, returns 202) |
| `GET`  | `/api/v1/documents` | List user's documents with status, doc_date, doc_type |
| `GET`  | `/api/v1/documents/{doc_id}` | Poll ingestion status |
| `DELETE` | `/api/v1/documents/{doc_id}` | Remove vectors + SQLite row + local files |

### Retrieval

| Method | Path | Returns | LLM? |
|---|---|---|---|
| `POST` | `/api/v1/search` | Ranked chunks + scores + source metadata + query_plan | No |
| `POST` | `/api/v1/ask` | Generated answer + inline `[n]` citations + references array | Yes (gpt-4o) |

### Utility

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Weaviate connection status |
| `GET` | `/api/v1/routes` | List all registered endpoints |
| `GET` | `/docs` | Interactive Swagger UI |

### `/search` request shape

```json
{
  "query": "what is the liability cap in the contract",
  "doc_type_filter": "legal",
  "k": 8,
  "alpha": 0.7,
  "score_threshold": 0.65
}
```

Returns: `chunks[]` with `rank`, `content`, `score`, `rerank_score`, `filename`,
`page_no`, `doc_type`, `chunk_type`, `doc_id`, `date`, `file_path`, `collection`.

### `/ask` request shape

```json
{
  "question": "Summarise the indemnification clauses across all uploaded contracts",
  "doc_type_filter": "legal",
  "k": 12,
  "score_threshold": 0.70
}
```

Returns: `answer` (markdown with inline `[1]` `[2]` citations), `references[]`
(filename, page_no, score, chunk_type, doc_id, date), `low_confidence`, `chunks_used`,
`chunks_filtered`, `router_reason`.

---

## Scale & Cost Estimates (Q3 supplementary)

### 100K documents → ~500K vectors

| Phase | Approach | Time estimate |
|---|---|---|
| Parsing | Async FastAPI background tasks | ~8 hrs (parallel ingestion, 100 docs/hr per worker) |
| Embedding (small, 400K chunks) | OpenRouter batch=100, async | ~4 hrs |
| Embedding (large, 100K legal chunks) | OpenRouter batch=100, async | ~2 hrs |
| Weaviate upsert (500K objects, 500/batch) | 1000 batch calls | ~30 min |

**What breaks first at 1M documents:**
1. Single Weaviate node → HNSW index exceeds RAM → switch to `pq` (product quantisation)
   compression or shard across nodes
2. SQLite → migrate to PostgreSQL (async SQLAlchemy already abstracts this; one
   connection string change)
3. OpenRouter rate limits on embedding calls → add exponential backoff + queue
   (Celery or ARQ)

**50 clients × 100K docs (5M documents total):**
Each client = one Weaviate tenant. Weaviate's multi-tenant architecture handles
10,000+ tenants; the physical limit is RAM for concurrent hot tenants. Cold tenants
(inactive users) have their indexes offloaded to disk automatically in Weaviate v1.24+.
Ingestion is fully parallelisable across tenants — no shared state.

**Rough monthly cost (at steady state, all clients active):**

| Item | Volume | Unit cost | Monthly |
|---|---|---|---|
| OpenRouter embedding (new docs) | 500K chunks/mo | $0.02/1M tokens (small) | ~$5 |
| OpenRouter routing (queries) | 100K queries | $0.15/1M tokens (gpt-4o-mini) | ~$2 |
| OpenRouter answers (queries) | 100K queries × 2K tokens | $2.50/1M tokens (gpt-4o) | ~$500 |
| Weaviate cloud (5M vectors) | 5M vectors | ~$0.05/1M vectors/mo | ~$250 |
| Local compute (reranker, CLIP) | CPU inference | EC2 t3.large | ~$60 |
| **Total** | | | **~$817/mo** |

LLM answer generation dominates cost. At query volume, caching common answers
(Redis, 1-hr TTL, keyed on embedding similarity) would cut LLM spend by ~40%.
