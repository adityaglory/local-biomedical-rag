import streamlit as st
import requests
import os
import json

st.set_page_config(page_title="Biomedical AI Assistant", page_icon="🧬")

st.title("🧬 GenAI Biomedis (RAG Local)")
st.caption("Powered by Llama 3.2, FastAPI, & LangChain")

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Tanya sesuatu tentang paper biomedis..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    try:
        with st.spinner("Sedang membaca jurnal..."):
            response = requests.post(
                f"{BACKEND_URL}/chat", 
                json={"text": prompt}
            )
            
            if response.status_code == 200:
                data = response.json()
                answer = data["answer"]
                sources = data["sources"]
                
                full_response = f"{answer}\n\n---\n**Sumber Referensi:**"
                for src in sources:
                    full_response += f"\n- {src}"
            else:
                full_response = "⚠️ Maaf, server backend lagi ngambek (Error API)."
                
    except Exception as e:
        full_response = f"❌ Gagal konek ke backend: {e}"

    with st.chat_message("assistant"):
        st.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})
