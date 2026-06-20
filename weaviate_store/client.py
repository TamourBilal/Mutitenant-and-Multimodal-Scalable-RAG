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


def delete_tenant(user_id: str) -> None:
    """Remove the user's tenant (and all its vectors) from ALL three collections."""
    client = get_weaviate_client()

    for coll_name in _ALL_COLLECTIONS:
        try:
            collection = client.collections.get(coll_name)
            if user_id in set(collection.tenants.get().keys()):
                collection.tenants.remove([user_id])
                logger.info("Removed Weaviate tenant | user_id=%s collection=%s", user_id, coll_name)
        except Exception as e:
            logger.warning("Failed removing tenant | user_id=%s collection=%s err=%s", user_id, coll_name, e)
