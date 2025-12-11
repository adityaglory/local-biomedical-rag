# rag_engine.py
import os
import warnings

# Sesuaikan import ini sama workaround lo tadi ya!
from langchain_classic.chains import RetrievalQA 
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings

warnings.filterwarnings("ignore")

class LocalRAG:
    def __init__(self, db_path="./chroma_db", llm_model="llama3.2"):
        # Kita load database & model SEKALI aja pas server nyala.
        # Ini jauh lebih efisien daripada load tiap kali ada pertanyaan.
        print("⚙️  Loading RAG Engine...")
        
        self.embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.db = Chroma(persist_directory=db_path, embedding_function=self.embedding_function)
        # self.retriever = self.db.as_retriever(search_kwargs={"k": 3})
        # TUNING PART 🔧
        # Kita ganti jadi MMR biar variasi konteksnya lebih luas
        # k=6: Kita kasih makan AI 6 potongan teks (sebelumnya cuma 3)
        # fetch_k=20: Cari 20 kandidat dulu, baru filter jadi 6 yang paling unik
        self.retriever = self.db.as_retriever(
            search_type="similarity", # Balik ke similarity biasa tapi jumlah banyak
            search_kwargs={"k": 12} 
        )
        
        ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        
        print(f"🔌 Connecting to Ollama at: {ollama_base_url}")
        
        self.llm = ChatOllama(
            model=llm_model, 
            temperature=0.3,
            base_url=ollama_base_url # <--- Tambahin parameter ini
        )
        
        template = """
        Anda adalah asisten peneliti biomedis. Jawab HANYA berdasarkan konteks.
        
        Konteks: {context}
        Pertanyaan: {question}
        Jawaban:
        """
        prompt = PromptTemplate.from_template(template)
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.retriever,
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )
        print("✅ RAG Engine Ready!")

    def ask(self, query: str):
        if not query:
            return None
        return self.qa_chain.invoke({"query": query})