import os
from pathlib import Path

# =====================================================================
# MODEL CONFIGURATIONS
# =====================================================================
# Local Ollama model (llama3 expected to be pre-installed)
LLM_MODEL = "llama3"

# HuggingFace embedding model for vector search
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# =====================================================================
# VECTOR DATABASE CONFIGURATION
# =====================================================================
CHROMA_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_db")

# Collection name for storing UTM knowledge base
CHROMA_COLLECTION_NAME = "myutm_knowledge_base"

# Chunking settings for document processing
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# =====================================================================
# INFERENCE SETTINGS
# =====================================================================
LLM_REQUEST_TIMEOUT = 60.0  # Timeout for Ollama requests (seconds)
LLM_TEMPERATURE = 0.0  # Deterministic output (no hallucinations)

# Greetings list -- single source of truth, imported by both app.py and inference.py
# so the two layers can never disagree on what counts as a greeting.
GREETINGS = ["hello", "hi", "hey", "assalamualaikum", "selamat datang", "selamat pagi"]

# =====================================================================
# RETRIEVAL SETTINGS
# =====================================================================
# Minimum similarity score a retrieved chunk needs to be treated as relevant.
# Without this, the retriever always returns its top-k closest chunks even when
# none of them are actually relevant, and the LLM has to notice that on its own.
# Tune this: print [r.score for r in results] for a few on-topic and off-topic
# test queries and pick a cutoff that separates them -- there's no universal
# "correct" number, it depends on your embedding model and your data.
SIMILARITY_CUTOFF = 0.45
# =====================================================================
# CONFIGURATION VALIDATION
# =====================================================================
def validate_config():
    """Validate that required directories and configurations exist."""
    # Ensure chroma_db parent directory exists
    db_parent = Path(CHROMA_DB_PATH).parent
    if not db_parent.exists():
        raise ValueError(f"Database parent directory does not exist: {db_parent}")
    
    # Create chroma_db if it doesn't exist
    Path(CHROMA_DB_PATH).mkdir(parents=True, exist_ok=True)

# Run validation on module load
try:
    validate_config()
except Exception as e:
    print(f"⚠️  Configuration validation warning: {e}")