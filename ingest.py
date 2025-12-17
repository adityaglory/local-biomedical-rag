import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

DATA_PATH = "./data"
DB_PATH = "./chroma_db"

def create_vector_db():
    print(f"🚀 Memulai proses Ingestion dari folder: {DATA_PATH}...")
    
    documents = []
    for filename in os.listdir(DATA_PATH):
        if filename.endswith(".pdf"):
            file_path = os.path.join(DATA_PATH, filename)
            print(f"   📄 Membaca file: {filename}")
            
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
            
    print(f"✅ Total halaman terbaca: {len(documents)}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    print(f"🧩 Dokumen dipecah menjadi: {len(chunks)} chunks")

    print("   🔮 Memuat model embedding (ini butuh internet bentar buat download modelnya)...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print("   💾 Menyimpan ke ChromaDB lokal...")
    db = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory=DB_PATH
    )
    
    print(f"🎉 Selesai! Database vektor tersimpan di {DB_PATH}")

if __name__ == "__main__":
    create_vector_db()
