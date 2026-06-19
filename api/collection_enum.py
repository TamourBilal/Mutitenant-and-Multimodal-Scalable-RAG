from __future__ import annotations

from enum import Enum
from typing import List, Optional

from config import settings


class Collection(str, Enum):
    documents = "documents"   # PDFs, text docs → RAGDocuments
    tables    = "tables"      # Table-heavy PDFs → RAGTables
    images    = "images"      # Images with captions → RAGImages
    html      = "html"        # HTML pages → RAGDocuments (same collection, filtered by doc_type)


_MAP = {
    Collection.documents: settings.WEAVIATE_COLLECTION_DOCUMENTS,
    Collection.tables:    settings.WEAVIATE_COLLECTION_TABLES,
    Collection.images:    settings.WEAVIATE_COLLECTION_IMAGES,
    Collection.html:      settings.WEAVIATE_COLLECTION_DOCUMENTS,
}

# doc_type filter applied automatically when collection=html
_DOC_TYPE_FILTER = {
    Collection.html: "html",
}

_DISPLAY = {
    Collection.documents: "documents",
    Collection.tables:    "tables",
    Collection.images:    "images",
    Collection.html:      "html",
}

_ALL = [
    settings.WEAVIATE_COLLECTION_DOCUMENTS,
    settings.WEAVIATE_COLLECTION_TABLES,
    settings.WEAVIATE_COLLECTION_IMAGES,
]


def resolve(collections: Optional[List[Collection]]) -> List[str]:
    """Convert Collection enums → unique Weaviate collection names."""
    if not collections:
        return _ALL
    seen = []
    for c in collections:
        name = _MAP[c]
        if name not in seen:
            seen.append(name)
    return seen


def doc_type_filter_for(collections: List[Collection]) -> Optional[str]:
    """Return a doc_type filter if ALL selected collections map to one type (e.g. html)."""
    filters = [_DOC_TYPE_FILTER[c] for c in collections if c in _DOC_TYPE_FILTER]
    if filters and len(filters) == len(collections):
        return filters[0]
    return None


def display_name(collection: Collection) -> str:
    return _DISPLAY.get(collection, collection.value)
