from __future__ import annotations

import logging

import weaviate
from weaviate.classes.config import Configure, DataType, Property, VectorDistances

from config import settings

logger = logging.getLogger(__name__)

# Named-vector combinations supported per collection:
#   RAGDocuments → text_small (1536) + text_large (3072)
#   RAGTables    → text_small (1536) + text_large (3072)
#   RAGImages    → text_small (1536) + image       (512)
COLLECTION_VECTORS: dict[str, list[str]] = {
    settings.WEAVIATE_COLLECTION_DOCUMENTS: ["text_small", "text_large"],
    settings.WEAVIATE_COLLECTION_TABLES:    ["text_small", "text_large"],
    settings.WEAVIATE_COLLECTION_IMAGES:    ["text_small", "image"],
}

_COMMON_PROPERTIES = [
    Property(name="content",     data_type=DataType.TEXT),    # BM25-indexed; chunk text / caption
    Property(name="source",      data_type=DataType.TEXT),    # pdf|html|image|other
    Property(name="doc_type",    data_type=DataType.TEXT),    # legal|pdf|html|image|other
    Property(name="date",        data_type=DataType.DATE),    # ingestion ISO-8601
    Property(name="doc_id",      data_type=DataType.TEXT),
    Property(name="chunk_index", data_type=DataType.INT),
    Property(name="page_no",     data_type=DataType.INT),     # 0 for non-PDF
    Property(name="file_path",   data_type=DataType.TEXT),    # local absolute path
    Property(name="filename",    data_type=DataType.TEXT),
    Property(name="chunk_type",  data_type=DataType.TEXT),    # text|table|image_caption
    Property(name="user_id",     data_type=DataType.TEXT),
]

_HNSW = Configure.VectorIndex.hnsw(distance_metric=VectorDistances.COSINE)


def _create_text_collection(client: weaviate.WeaviateClient, name: str) -> None:
    """text_small + text_large named vectors — for RAGDocuments and RAGTables."""
    client.collections.create(
        name=name,
        multi_tenancy_config=Configure.multi_tenancy(enabled=True),
        vectorizer_config=[
            Configure.NamedVectors.none(name="text_small", vector_index_config=_HNSW),
            Configure.NamedVectors.none(name="text_large", vector_index_config=_HNSW),
        ],
        properties=_COMMON_PROPERTIES,
    )
    logger.info("Created Weaviate collection | name=%s vectors=text_small+text_large", name)


def _create_image_collection(client: weaviate.WeaviateClient, name: str) -> None:
    """text_small (caption) + image (CLIP 512-dim) named vectors — for RAGImages."""
    client.collections.create(
        name=name,
        multi_tenancy_config=Configure.multi_tenancy(enabled=True),
        vectorizer_config=[
            Configure.NamedVectors.none(name="text_small", vector_index_config=_HNSW),
            Configure.NamedVectors.none(name="image",      vector_index_config=_HNSW),
        ],
        properties=_COMMON_PROPERTIES,
    )
    logger.info("Created Weaviate collection | name=%s vectors=text_small+image", name)


def ensure_schema(client: weaviate.WeaviateClient) -> None:
    """Create RAGDocuments, RAGTables, RAGImages collections if they don't exist."""
    existing = set(client.collections.list_all().keys())

    for name in (settings.WEAVIATE_COLLECTION_DOCUMENTS, settings.WEAVIATE_COLLECTION_TABLES):
        if name not in existing:
            _create_text_collection(client, name)
        else:
            logger.info("Weaviate collection already exists | name=%s", name)

    if settings.WEAVIATE_COLLECTION_IMAGES not in existing:
        _create_image_collection(client, settings.WEAVIATE_COLLECTION_IMAGES)
    else:
        logger.info("Weaviate collection already exists | name=%s", settings.WEAVIATE_COLLECTION_IMAGES)
