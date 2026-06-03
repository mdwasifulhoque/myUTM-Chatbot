import os

# Model Configurations
LLM_MODEL = "llama3"
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# Vector Database Path
CHROMA_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_db")