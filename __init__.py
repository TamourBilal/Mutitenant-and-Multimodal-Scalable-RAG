"""
RAG module exports for multi-tenant Pinecone vector database.

Heavy imports (PDF parser with PyMuPDF) are lazy so agent / retrieval paths
do not require ``fitz`` unless PDF parsing is used.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from core_scripts.rag.docuement_retrieval import VectorDBManager
from core_scripts.rag.document_ingestion import PdfDocumentIngestion, UserProductIngestion
from core_scripts.rag.metadata_utils import chunk_provenance
from core_scripts.rag.pdf_format_convert import PDFConverter
from core_scripts.rag.text_splitter import PdfMarkdownTextSplitter

if TYPE_CHECKING:
    from core_scripts.rag.document_parser import PDFProcessorAsync

__all__ = [
    "PdfMarkdownTextSplitter",
    "PDFProcessorAsync",
    "VectorDBManager",
    "PdfDocumentIngestion",
    "UserProductIngestion",
    "PDFConverter",
    "chunk_provenance",
]


def __getattr__(name: str):
    if name == "PDFProcessorAsync":
        from core_scripts.rag.document_parser import PDFProcessorAsync

        return PDFProcessorAsync
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
