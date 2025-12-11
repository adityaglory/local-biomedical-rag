# 🧬 Local Biomedical RAG (Dockerized)

A privacy-first, local Retrieval-Augmented Generation (RAG) system designed to analyze and query biomedical research papers offline.

Built with an End-to-End MLOps mindset, this project decouples the inference engine (Backend) from the user interface (Frontend) using Docker containers, ensuring reproducibility and scalability.

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
   ```
   
### Installation & Run
Clone this repository and launch the containerized stack:
```bash
docker compose up --build
```
The system will spin up two services:
- Frontend Dashboard: http://localhost:8501
- Backend API Docs: http://localhost:8000/docs

## 📂 Project Structure

├── api.py           # FastAPI entry point

├── dashboard.py     # Streamlit frontend logic

├── rag_engine.py    # Core RAG logic (Chain, Retrieval, Prompting)

├── Dockerfile       # Container definition

├── docker-compose.yml # Service orchestration

├── requirements.txt # Python dependencies

└── data/            # Source PDF documents

## 🧠 Key Feature
- Context-Aware Retrieval: Uses MMR/Similarity search with k=12 to handle long-context biomedical queries (e.g., extracting specific accuracy metrics from results sections).
- Microservices Architecture: Frontend and Backend communicate via Docker internal networking.
- Dockerized Environment: Eliminates "it works on my machine" issues.


