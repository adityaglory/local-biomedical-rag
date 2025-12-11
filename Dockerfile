# 1. Pilih Base Image (OS dasar)
# Kita pake Python 3.11 versi 'slim' biar ukurannya kecil & ringan
FROM python:3.11-slim

# 2. Set Folder Kerja
# Semua file project lo bakal ditaruh di folder /app di dalam container
WORKDIR /app

# 3. Install System Dependencies
# ChromaDB kadang butuh tools C++ buat compile (build-essential)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy & Install Library Python
# Kita copy requirements.txt duluan biar Docker bisa nge-cache (biar build ulang cepet)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy Seluruh Kode Project
# Masukin api.py, rag_engine.py, dashboard.py, folder data, dll ke dalam container
COPY . .

# 6. Buka Port (Pintu Akses)
# Port 8000 buat API Backend
# Port 8501 buat Dashboard Streamlit
EXPOSE 8000 8501

# 7. Command Default
# Perintah yang jalan pertama kali (nanti ini bisa di-override sama docker-compose)
CMD ["python", "api.py"]