from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import warnings

# Biar output terminal bersih dari warning yang tadi
warnings.filterwarnings("ignore")

# 1. SETUP CONFIG ⚙️
# Pastikan path-nya sama persis dengan yang di ingest.py
DB_PATH = "./chroma_db"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def test_retrieval():
    print("🔍 Memuat database vektor...")
    
    # Load model embedding yang SAMA PERSIS dengan waktu ingest
    embedding_function = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    
    # Load database Chroma
    db = Chroma(persist_directory=DB_PATH, embedding_function=embedding_function)
    
    # 2. TEST QUERY ❓
    # Kita tanya sesuatu yang teknis dari paper ke-2 (CNN-for-Hand-Washing...)
    query = "How accurate is MobileNetV2 on Kaggle dataset?"
    print(f"\n❓ Pertanyaan: '{query}'")
    
    # Lakukan pencarian (Similarity Search)
    # k=3 artinya kita minta 3 potongan teks paling relevan
    results = db.similarity_search(query, k=3)
    
    print(f"\n✅ Ditemukan {len(results)} referensi relevan:\n")
    
    for i, doc in enumerate(results, 1):
        # Ambil nama file asalnya
        source = doc.metadata.get("source", "Unknown")
        # Ambil isi teksnya (gw potong dikit biar gak kepanjangan di terminal)
        content = doc.page_content[:300].replace("\n", " ")
        
        print(f"--- [Referensi {i}] ---")
        print(f"📄 Sumber: {source}")
        print(f"📝 Isi: {content}...")
        print("-" * 40)

if __name__ == "__main__":
    test_retrieval()