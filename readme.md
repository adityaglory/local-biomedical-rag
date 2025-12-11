# 🧬 Local Biomedical RAG (Dockerized)

A privacy-first, local Retrieval-Augmented Generation (RAG) system designed to analyze and query biomedical research papers offline.

Built with an **End-to-End MLOps** mindset, this project decouples the inference engine (Backend) from the user interface (Frontend) using Docker containers, ensuring reproducibility and scalability.

## 🏗️ Architecture
- **Inference Engine:** Llama 3.2 (via Ollama)
- **Vector Store:** ChromaDB (Persistence enabled)
- **Backend API:** FastAPI (Async)
- **Frontend UI:** Streamlit
- **Orchestration:** Docker Compose
- **Retrieval Strategy:** MMR (Maximal Marginal Relevance) for diverse context fetching.

## 🚀 Quick Start

### Prerequisites
1. **Docker Desktop** installed and running (WSL 2 integration enabled for Windows users).
2. **Ollama** installed on the host machine.
3. Pull the required LLM model:
   ```bash
   ollama pull llama3.2
