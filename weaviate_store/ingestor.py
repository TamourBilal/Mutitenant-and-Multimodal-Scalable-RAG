from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from config import settings
from weaviate_store.client import get_weaviate_client

logger = logging.getLogger(__name__)

# Route chunk_type → Weaviate collection
_CHUNK_COLLECTION: Dict[str, str] = {
    "text":          settings.WEAVIATE_COLLECTION_DOCUMENTS,
    "table":         settings.WEAVIATE_COLLECTION_TABLES,
    "image_caption": settings.WEAVIATE_COLLECTION_IMAGES,
}

_ALL_COLLECTIONS = [
    settings.WEAVIATE_COLLECTION_DOCUMENTS,
    settings.WEAVIATE_COLLECTION_TABLES,
    settings.WEAVIATE_COLLECTION_IMAGES,
]


def _collection_for(chunk_type: str) -> str:
    return _CHUNK_COLLECTION.get(chunk_type, settings.WEAVIATE_COLLECTION_DOCUMENTS)


@dataclass
class WeaviateObject:
    """A single chunk ready for upsert."""
    properties: Dict[str, Any]
    # Only the named vector(s) applicable to this chunk type are populated
    vector_text_small: Optional[List[float]] = None   # 1536-dim: text / captions
    vector_text_large: Optional[List[float]] = None   # 3072-dim: legal text / tables
    vector_image: Optional[List[float]] = None         # 512-dim:  CLIP image embeddings


def _build_named_vector(obj: WeaviateObject) -> Dict[str, List[float]]:
    vec: Dict[str, List[float]] = {}
    if obj.vector_text_small:
        vec["text_small"] = obj.vector_text_small
    if obj.vector_text_large:
        vec["text_large"] = obj.vector_text_large
    if obj.vector_image:
        vec["image"] = obj.vector_image
    return vec


def _sync_upsert(objects: List[WeaviateObject], user_id: str) -> Dict[str, Any]:
    """Group by collection then batch-insert — runs in a thread."""
    from weaviate.classes.data import DataObject

    client = get_weaviate_client()

    # Group objects by target collection
    groups: Dict[str, List[WeaviateObject]] = defaultdict(list)
    for obj in objects:
        coll = _collection_for(obj.properties.get("chunk_type", "text"))
        groups[coll].append(obj)

    total = 0
    errors: List[str] = []

    for coll_name, coll_objects in groups.items():
        collection = client.collections.get(coll_name)
        tenant_col = collection.with_tenant(user_id)

        for i in range(0, len(coll_objects), settings.WEAVIATE_BATCH_SIZE):
            batch_slice = coll_objects[i : i + settings.WEAVIATE_BATCH_SIZE]
            weaviate_objects = []

            for obj in batch_slice:
                vec = _build_named_vector(obj)
                if not vec:
                    logger.warning(
                        "Object has no vector, skipping | doc_id=%s chunk_type=%s",
                        obj.properties.get("doc_id"),
                        obj.properties.get("chunk_type"),
                    )
                    continue
                weaviate_objects.append(DataObject(properties=obj.properties, vector=vec))

            if not weaviate_objects:
                continue

            result = tenant_col.data.insert_many(weaviate_objects)

            batch_errors = (
                result.errors if hasattr(result, "errors") and result.errors else {}
            )
            for idx, err in batch_errors.items():
                msg = f"{coll_name}[{i + idx}]: {err}"
                errors.append(msg)
                logger.error("[INGESTOR] Upsert error | %s", msg)

            batch_ok = len(weaviate_objects) - len(batch_errors)
            total += batch_ok
            logger.info(
                "[INGESTOR] Batch upserted | collection=%s ok=%d errors=%d offset=%d user=%s",
                coll_name, batch_ok, len(batch_errors), i, user_id,
            )

    return {"upserted": total, "errors": len(errors), "error_details": errors[:10]}


async def batch_upsert(objects: List[WeaviateObject], user_id: str) -> Dict[str, Any]:
    """Async wrapper: runs Weaviate batch upsert in a thread pool."""
    if not objects:
        return {"upserted": 0, "errors": 0, "error_details": []}
    logger.info("[INGESTOR] Starting upsert | objects=%d user=%s", len(objects), user_id)
    return await asyncio.to_thread(_sync_upsert, objects, user_id)


def _sync_delete_by_doc(doc_id: str, user_id: str) -> Dict[str, Any]:
    """Delete all vectors for a document from every collection."""
    from weaviate.classes.query import Filter

    client = get_weaviate_client()
    for coll_name in _ALL_COLLECTIONS:
        try:
            collection = client.collections.get(coll_name)
            tenant_col = collection.with_tenant(user_id)
            tenant_col.data.delete_many(
                where=Filter.by_property("doc_id").equal(doc_id)
            )
            logger.info(
                "[INGESTOR] Deleted vectors | collection=%s doc_id=%s user=%s",
                coll_name, doc_id, user_id,
            )
        except Exception as exc:
            logger.warning(
                "[INGESTOR] Delete failed | collection=%s doc_id=%s err=%s",
                coll_name, doc_id, exc,
            )

    return {"status": "deleted", "doc_id": doc_id}


async def delete_document_vectors(doc_id: str, user_id: str) -> Dict[str, Any]:
    return await asyncio.to_thread(_sync_delete_by_doc, doc_id, user_id)
