import streamlit as st
import requests
import os
import json

# Konfigurasi Halaman
st.set_page_config(page_title="Biomedical AI Assistant", page_icon="🧬")

st.title("🧬 GenAI Biomedis (RAG Local)")
st.caption("Powered by Llama 3.2, FastAPI, & LangChain")

# TENTUKAN URL BACKEND 🔗
# Kalau di Docker, dia bakal pake variabel environment 'BACKEND_URL'
# Kalau di Laptop biasa, dia bakal default ke 'http://localhost:8000'
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# Inisialisasi Session State buat simpen chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Tampilkan chat history yang lama
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input user baru
if prompt := st.chat_input("Tanya sesuatu tentang paper biomedis..."):
    # 1. Tampilkan pertanyaan user
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. Kirim ke Backend API lo (FastAPI)
    try:
        with st.spinner("Sedang membaca jurnal..."):
            # GANTI HARDCODED URL DENGAN VARIABLE
            response = requests.post(
                f"{BACKEND_URL}/chat", # <--- Pake f-string
                json={"text": prompt}
            )
            
            if response.status_code == 200:
                data = response.json()
                answer = data["answer"]
                sources = data["sources"]
                
                # Format jawaban + sumber
                full_response = f"{answer}\n\n---\n**Sumber Referensi:**"
                for src in sources:
                    full_response += f"\n- {src}"
            else:
                full_response = "⚠️ Maaf, server backend lagi ngambek (Error API)."
                
    except Exception as e:
        full_response = f"❌ Gagal konek ke backend: {e}"

    # 3. Tampilkan jawaban AI
    with st.chat_message("assistant"):
        st.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})