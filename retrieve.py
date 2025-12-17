from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import warnings

warnings.filterwarnings("ignore")

# 1. SETUP CONFIG ⚙️
# Pastikan path sama persis dengan di ingest.py
DB_PATH = "./chroma_db"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def test_retrieval():
    print("🔍 Memuat database vektor...")
    
    # Load model embedding yang SAMA PERSIS dengan waktu ingest
    embedding_function = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    
    # Load database Chroma
    db = Chroma(persist_directory=DB_PATH, embedding_function=embedding_function)
    
    # 2. TEST QUERY ❓
    query = "How accurate is MobileNetV2 on Kaggle dataset?"
    print(f"\n❓ Pertanyaan: '{query}'")
    results = db.similarity_search(query, k=3)
    
    print(f"\n✅ Ditemukan {len(results)} referensi relevan:\n")
    
    for i, doc in enumerate(results, 1):
        # Ambil nama file asalnya
        source = doc.metadata.get("source", "Unknown")
        content = doc.page_content[:300].replace("\n", " ")
        
        print(f"--- [Referensi {i}] ---")
        print(f"📄 Sumber: {source}")
        print(f"📝 Isi: {content}...")
        print("-" * 40)

if __name__ == "__main__":
    test_retrieval()
