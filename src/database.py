import os
import logging
import warnings

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
logging.getLogger("transformers").setLevel(logging.ERROR)

import chromadb
from llama_index.core import Settings, StorageContext, VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

# 1. Define Local Paths
DB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_db")
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def initialize_rag_settings():
    """
    Configures global LlamaIndex settings to use the local BAAI embedding model.
    This downloads the weights locally on the first run.
    """
    print("Initializing local embedding model (BAAI/bge-small-en-v1.5)...")
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    # Set globally across LlamaIndex orchestrations
    Settings.embed_model = embed_model
    # We will hook up the Ollama LLM component when linking inference.py later


def get_vector_index():
    """
    Connects to the persistent local ChromaDB instance.
    If data exists, it loads the index. If empty, it returns None.
    """
    # Initialize global settings first
    initialize_rag_settings()

    # Initialize persistent disk client for ChromaDB
    db_client = chromadb.PersistentClient(path=DB_DIR)

    # Create or fetch our specialized UTM document collection
    chroma_collection = db_client.get_or_create_collection("myutm_knowledge_base")

    # Construct the vector store layer
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Check if collection already contains vectors to prevent redundant reprocessing
    if chroma_collection.count() > 0:
        print(f"Loading existing vector database index ({chroma_collection.count()} vectors found)...")
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            storage_context=storage_context
        )
        return index
    else:
        print("Vector database is currently empty.")
        return None


def build_or_refresh_database():
    """
    Emergency ingestion fallback helper. Reads raw files inside the /data directory,
    chunks them, runs embeddings, and saves them to disk.
    """
    initialize_rag_settings()

    if not os.path.exists(DATA_DIR) or not os.listdir(DATA_DIR):
        print(f"CRITICAL: Put your source files (PDFs/txt) inside the '{DATA_DIR}' folder first!")
        return None

    print(f"Reading target documentation from: {DATA_DIR}...")
    # Read files natively via LlamaIndex's SimpleDirectoryReader (handles PDF/TXT out-of-the-box via pypdf)
    documents = SimpleDirectoryReader(DATA_DIR).load_data()

    # Set explicit chunk limits to fit safely inside Llama 3's context boundaries
    Settings.chunk_size = 500
    Settings.chunk_overlap = 50

    print("Chunking documents, running embeddings, and indexing into ChromaDB (this might take a second)...")
    db_client = chromadb.PersistentClient(path=DB_DIR)
    chroma_collection = db_client.get_or_create_collection("myutm_knowledge_base")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # This automatically pushes data into the persistent disk folder
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True
    )
    print("Database built successfully and locked on disk!")
    return index


if __name__ == "__main__":
    # If run directly, test or build database
    idx = get_vector_index()
    if idx is None:
        build_or_refresh_database()