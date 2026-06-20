"""
myUTM Chatbot Source Package
Provides RAG-based inference engine and vector database utilities.
"""

from .inference import UTMInferenceEngine
from .database import get_vector_index, build_or_refresh_database
from .config import (
    LLM_MODEL, 
    EMBED_MODEL_NAME, 
    CHROMA_DB_PATH, 
    CHROMA_COLLECTION_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    validate_config
)

__all__ = [
    "UTMInferenceEngine",
    "get_vector_index",
    "build_or_refresh_database",
    "LLM_MODEL",
    "EMBED_MODEL_NAME",
    "CHROMA_DB_PATH",
    "CHROMA_COLLECTION_NAME",
    "CHUNK_SIZE",
    "CHUNK_OVERLAP",
    "validate_config",
]
