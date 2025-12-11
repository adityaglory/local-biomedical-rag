# api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag_engine import LocalRAG # <-- Kita import class yang tadi dibuat

app = FastAPI(title="Biomedical RAG API", version="1.0")

# Inisialisasi RAG Engine pas server mulai
# Jadi dia standby terus di memori
rag_engine = LocalRAG()

# Format data request (biar validasi otomatis)
class QueryRequest(BaseModel):
    text: str

@app.get("/")
def read_root():
    return {"status": "active", "service": "Local RAG Biomedis"}

@app.post("/chat")
def chat_endpoint(request: QueryRequest):
    try:
        # Panggil fungsi ask dari rag_engine
        response = rag_engine.ask(request.text)
        
        # Rapikan sumber referensi
        sources = []
        for doc in response["source_documents"]:
            src = doc.metadata.get('source', 'Unknown').split('/')[-1]
            page = doc.metadata.get('page', 0)
            sources.append(f"{src} (Hal. {page})")
            
        return {
            "question": request.text,
            "answer": response["result"],
            "sources": sources
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Jalanin server di localhost port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)