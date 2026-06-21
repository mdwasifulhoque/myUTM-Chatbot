# myUTM Intelligent Assistant 🤖🏛️

A locally-hosted Retrieval-Augmented Generation (RAG) chatbot built to help Universiti Teknologi Malaysia (UTM) students get quick, accurate answers on campus logistics — shuttle bus schedules, academic calendar dates, registration FAQs, and student services contacts.

Built as an MVP for UTM's AI Showcase.

---

## Overview

myUTM Intelligent Assistant answers student questions by retrieving relevant passages from a curated set of official UTM documents and grounding a local LLM's response in that retrieved context — rather than relying on the model's general knowledge. This keeps answers traceable to source documents and avoids confidently-wrong responses on topics the knowledge base doesn't cover.

The entire system — language model, embedding model, and vector database — runs locally via Ollama and ChromaDB. No student query or document content is sent to an external API.

## Features

- **Conversational chat interface** built with Streamlit, including quick-access buttons for common questions (shuttle bus times, library hours, exam schedules)
- **Retrieval-augmented responses** grounded in official UTM documentation, with a relevance threshold to avoid answering from irrelevant retrieved chunks
- **Scoped refusal behavior** — declines requests outside campus logistics (code generation, essays, general knowledge) and resists common prompt-injection patterns
- **Bilingual greeting handling** (English and Malay)
- **Fully local inference** — no external API calls for the LLM or embeddings

## Architecture

| Component | Technology |
|---|---|
| LLM Engine | [Ollama](https://ollama.com) running `llama3` (configurable in `src/config.py`) |
| Orchestration | [LlamaIndex](https://www.llamaindex.ai/) |
| Vector Database | [ChromaDB](https://www.trychroma.com/) (persistent, on-disk) |
| Embedding Model | `BAAI/bge-small-en-v1.5` (HuggingFace, runs locally via `sentence-transformers`) |
| Application Interface | [Streamlit](https://streamlit.io/) |

## Project Structure

```
myUTM-Chatbot/
├── data/                   # Knowledge base source documents (see below)
├── logo/
│   └── utm_logo.png
├── src/
│   ├── __init__.py
│   ├── config.py           # Models, paths, chunking, and retrieval settings
│   ├── database.py         # Builds/loads the ChromaDB vector index
│   └── inference.py        # RAG prompt construction and Ollama inference
├── app.py                  # Streamlit chat application
├── requirements.txt
└── README.md
```

## Prerequisites

- Python 3.10–3.13 (developed and tested on 3.13.4)
- [Ollama](https://ollama.com) installed locally, with the `llama3` model pulled
- ~5 GB free disk space for the LLM weights, plus space for the embedding model on first run

## Setup

**1. Clone the repository and enter the project directory**
```bash
git clone <repository-url>
cd myUTM-Chatbot
```

**2. Create and activate a virtual environment**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Install and prepare Ollama**

Download Ollama from [ollama.com](https://ollama.com), then pull the model:
```bash
ollama pull llama3
```
*(Optional)* To control where Ollama stores model weights, set this before pulling — replace the path with your own storage location:
```powershell
$env:OLLAMA_MODELS="C:\path\to\your\preferred\storage"
```

**5. Build the knowledge base**

`/data` is excluded from version control (see [Knowledge Base](#knowledge-base) below) — make sure it's populated before running this step, or the build will have nothing to index.

This reads everything in `/data`, chunks it, generates embeddings, and persists the index to `chroma_db/`. Run this once initially, and again any time files in `/data` change:
```bash
python -m src.database
```

**6. Launch the application**
```bash
streamlit run app.py
```
The app opens at `http://localhost:8501`.

## Configuration

Key tunables live in `src/config.py`:

| Setting | Purpose |
|---|---|
| `LLM_MODEL` | Which Ollama model to use |
| `EMBED_MODEL_NAME` | HuggingFace embedding model for retrieval |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | Document chunking parameters (in tokens) |
| `SIMILARITY_CUTOFF` | Minimum relevance score for a retrieved chunk to be used — filters out off-topic matches |
| `LLM_TEMPERATURE` | Set to `0.0` for deterministic, factual responses |

## Knowledge Base

`/data` is excluded from version control via `.gitignore` — since this is a public repository, the source documents (some of which include UTM staff contact details) aren't tracked here. **If you're a team member setting this up**, get the current document set from [wherever your team shares it — e.g. shared drive, internal channel] and place the files in `/data` before running the build step above.

The folder currently includes:
- 8 official UTM FAQ documents (registration, bursary, matric card, MySiswa card, residential colleges, transportation, UTMID, student welfare)
- Academic calendar (full PDF, plus a hand-curated quick-reference summary of key semester dates)
- On-campus shuttle bus schedules for UTM Johor Bahru and UTM Kuala Lumpur (reformatted as structured plain text — the original schedule PDFs use table/scan layouts that don't extract reliably as plain text)
- A directory of student services contacts

When adding new source documents, prefer plain text or well-structured PDFs where possible — tables, flowcharts, and scanned/image-based PDFs generally need to be reformatted into plain text before they retrieve reliably.

## Known Limitations

- Designed as a single-user, local MVP — inference runs through one local Ollama instance and isn't built for concurrent production traffic.
- Answers are scoped strictly to the documents in `/data`; the assistant is intentionally designed to decline rather than guess on anything outside that knowledge base.
- Retrieval quality depends on `SIMILARITY_CUTOFF`, which is tuned for the current embedding model and document set — re-tune if either changes significantly.

## License

*(Add your license here — e.g., academic/internal use only, MIT, etc.)*