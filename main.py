import warnings
import sys
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings

warnings.filterwarnings("ignore")

# 1. SETUP CONFIG ⚙️
DB_PATH = "./chroma_db"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama3.2" 

def start_rag_app():
    print("🚀 Menyiapkan RAG System...")
    
    try:
        # A. Load Database Vektor (Memory)
        print("   📂 Memuat memori (Vector DB)...")
        embedding_function = HuggingFaceEmbeddings(model_name=MODEL_NAME)
        db = Chroma(persist_directory=DB_PATH, embedding_function=embedding_function)
        retriever = db.as_retriever(search_kwargs={"k": 3}) 
        
        # B. Setup LLM (Brain)
        print(f"   🧠 Menghubungkan ke Ollama ({LLM_MODEL})...")
        llm = ChatOllama(model=LLM_MODEL, temperature=0.3) 

        # C. Prompt Engineering
        template = """
        Anda adalah asisten peneliti biomedis. Jawab pertanyaan HANYA berdasarkan konteks berikut. 
        Jika info tidak ada, katakan "Informasi tidak ditemukan di dokumen."
        
        Konteks:
        {context}
        
        Pertanyaan: 
        {question}
        
        Jawaban:
        """
        
        prompt = PromptTemplate.from_template(template)

        # D. Build Chain (Pipeline)
        print("   ⛓️  Menyusun pipeline AI...")
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True 
        )
        
        print("\n🤖 Sistem Siap! (Ketik 'exit' untuk keluar)")
        
        # E. Loop Tanya Jawab
        while True:
            query = input("\n🗣️ Kamu: ")
            if query.lower() in ["exit", "quit"]:
                break
            if not query:
                continue
                
            print("⏳ Sedang berpikir...")
            result = qa_chain.invoke({"query": query})
            
            print(f"\n🤖 AI: {result['result']}")
            print("\n📚 Sumber:")
            for doc in result["source_documents"]:
                src = doc.metadata.get('source', 'Unknown').split('/')[-1]
                print(f"- {src}")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("💡 Tips: Coba jalankan 'pip install -U langchain langchain-huggingface'")

if __name__ == "__main__":
    start_rag_app()
