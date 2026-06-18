from __future__ import annotations

import logging
from typing import Optional

import weaviate

from config import settings

logger = logging.getLogger(__name__)

_client: Optional[weaviate.WeaviateClient] = None

_ALL_COLLECTIONS = [
    settings.WEAVIATE_COLLECTION_DOCUMENTS,
    settings.WEAVIATE_COLLECTION_TABLES,
    settings.WEAVIATE_COLLECTION_IMAGES,
]


def get_weaviate_client() -> weaviate.WeaviateClient:
    global _client
    if _client is None or not _client.is_connected():
        _client = weaviate.connect_to_local(
            host=settings.WEAVIATE_HOST,
            port=settings.WEAVIATE_PORT,
            grpc_port=settings.WEAVIATE_GRPC_PORT,
        )
        logger.info(
            "Weaviate connected | host=%s port=%s",
            settings.WEAVIATE_HOST,
            settings.WEAVIATE_PORT,
        )
    return _client


def close_weaviate_client() -> None:
    global _client
    if _client and _client.is_connected():
        _client.close()
        _client = None
        logger.info("Weaviate client closed")


def ensure_tenant(user_id: str) -> None:
    """Create a Weaviate tenant for the user in ALL three collections if it doesn't exist."""
    from weaviate.classes.tenants import Tenant

    client = get_weaviate_client()

    for coll_name in _ALL_COLLECTIONS:
        collection = client.collections.get(coll_name)
        existing = set(collection.tenants.get().keys())
        if user_id not in existing:
            collection.tenants.create([Tenant(name=user_id)])
            logger.info("Created Weaviate tenant | user_id=%s collection=%s", user_id, coll_name)
