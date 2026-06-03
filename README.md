# myUTM Intelligent Assistant 🤖🏛️

A localized, secure Retrieval-Augmented Generation (RAG) chat application built to assist Universiti Teknologi Malaysia (UTM) students with campus services, library operations, and academic schedules.

---

## 🏗️ AI/ML Architecture Blueprint

This application is designed as a **100% Local-First AI System** to guarantee data privacy, zero latency data processing, and offline operational compliance.

* **Core LLM Engine:** Ollama running localized `llama3` weights (~4.7GB).
* **Context Strategy:** Retrieval-Augmented Generation (RAG) framework providing curated university documentation directly into the model inference window.
* **Application Interface:** Streamlit Engine providing asynchronous user chat state tracking.

---

## 🛠️ Local System Engineering & Installation

### Prerequisites
* Python 3.10 or higher
* Ollama (Windows Desktop Instance Engine)

### 1. Model Environment Initialization
Before launching, ensure your local Ollama instance points explicitly to your dedicated storage path and executes securely under localized CPU fallback rules if necessary:

```powershell
$env:OLLAMA_MODELS="D:\Softwares\Ollama"
ollama run llama3
