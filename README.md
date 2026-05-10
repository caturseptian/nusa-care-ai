# 🇮🇩 NusaCare AI

**🌐 Demo langsung:** <https://nusa-care-ai.streamlit.app>

**NusaCare AI** adalah chatbot layanan pelanggan berbahasa Indonesia untuk platform layanan publik digital fiktif **NusaCare**. Aplikasi ini membantu pengguna bertanya seputar layanan, prosedur, persyaratan, dan FAQ dalam Bahasa Indonesia.

Dibangun dengan **Streamlit + Google Gemini + ChromaDB** menggunakan pola **Retrieval-Augmented Generation (RAG)**.

---

## ✨ Fitur Utama

- 💬 **Antarmuka chatbot** sederhana berbasis Streamlit, lengkap dengan riwayat percakapan.
- 🧠 **Google Gemini** sebagai LLM utama (default model: `gemini-2.5-flash`).
- 🗂️ **RAG dengan ChromaDB** untuk menjawab berdasarkan basis pengetahuan resmi NusaCare.
- 🧬 **Embedding Gemini** (`gemini-embedding-001`) untuk vektor dokumen dan query.
- 🧰 **Sidebar konfigurasi**: pilih penyedia model, model spesifik, parameter generasi (temperature, top_k, top_p, max_output_tokens), jumlah konteks RAG, dan jumlah giliran riwayat.
- 🦙 **Ollama (opsional)** sebagai penyedia LLM lokal alternatif.
- 🌐 **Exa Web Search (opsional)** sebagai sumber konteks tambahan jika `EXA_API_KEY` diatur.
- 📚 **Tombol** Bangun Ulang Indeks, Reset Indeks, dan Bersihkan Riwayat.
- 🛡️ **Penanganan galat** ramah pengguna dalam Bahasa Indonesia (kunci API tidak valid, kuota, model tidak ditemukan, dan koneksi).
- 🇮🇩 **Seluruh UI dan jawaban** dalam Bahasa Indonesia.

---

## 🏛️ Arsitektur

```
Pengguna ──► Streamlit (app.py)
                │
                ├── Sidebar konfigurasi (src/config.py)
                ├── Memori sesi          (src/memory.py)
                ├── Retriever RAG        (src/rag.py)         ──► ChromaDB
                ├── Pencarian Exa (opsional, src/utils.py)    ──► Exa API
                └── Klien LLM            (src/gemini_client.py)
                                                │
                                                ├── Gemini (default)
                                                └── Ollama (opsional)
```

**Alur permintaan**:
1. Pengguna mengirim pertanyaan via `st.chat_input`.
2. Pertanyaan di-embed lalu dilakukan similarity search di ChromaDB → diambil top-k chunk.
3. (Opsional) Exa Web Search dipanggil untuk konteks tambahan.
4. System prompt + riwayat singkat + konteks + pertanyaan dikirim ke Gemini/Ollama.
5. Jawaban distream kembali ke UI dan disimpan di memori sesi.

---

## 📋 Prasyarat

- **Python ≥ 3.10**
- Kunci API **Google Gemini** (gratis di [Google AI Studio](https://aistudio.google.com/app/apikey)).
- (Opsional) **Ollama** terpasang lokal jika ingin penyedia alternatif: <https://ollama.com/download>.
- (Opsional) Kunci API **Exa** untuk pencarian web: <https://exa.ai>.

---

## ⚙️ Instalasi

```bash
# 1. Clone repo dan masuk ke direktori
git clone <url-repository> nusa-care-ai
cd nusa-care-ai

# 2. Buat virtual environment
python -m venv .venv
source .venv/bin/activate     # macOS/Linux
# .venv\Scripts\activate      # Windows

# 3. Pasang dependensi
pip install -r requirements.txt

# 4. Salin file environment dan isi GEMINI_API_KEY
cp .env.example .env
# lalu edit file .env, isi GEMINI_API_KEY dengan kunci Anda
```

> ⚠️ **Penting**: Jangan pernah meng-commit file `.env`. File `.gitignore` sudah mengabaikannya secara default.

---

## ▶️ Menjalankan Aplikasi

```bash
streamlit run app.py
```

Aplikasi akan terbuka di browser pada alamat <http://localhost:8501>.

**Saat pertama kali dijalankan**:
1. Pastikan status di sidebar menampilkan **"✅ Gemini API: siap"**.
2. Klik tombol **🔁 Bangun Ulang Indeks** di sidebar untuk memuat `data/knowledge_base.txt` ke ChromaDB.
3. Setelah indeks siap, status akan menjadi **"siap (N chunk)"**.
4. Mulai bertanya di kotak chat di bawah.

---

## 🧪 Contoh Pertanyaan

Coba beberapa pertanyaan ini untuk menguji chatbot:

- "Bagaimana cara mendaftar akun NusaCare?"
- "Berapa lama proses verifikasi identitas?"
- "Apa saja syarat untuk mengajukan beasiswa NusaCerdas?"
- "Bagaimana cara perpanjangan KTP via NusaCare?"
- "Apakah NusaCare berbayar?"
- "Bagaimana cara melaporkan penipuan yang mengatasnamakan NusaCare?"
- "Berapa jam operasional customer support?"
- "Bagaimana cara mereset kata sandi jika lupa?"

Pertanyaan di luar topik (mis. trivia umum atau permintaan data sensitif) akan ditolak dengan sopan oleh bot, sesuai aturan yang ditanamkan dalam *system prompt*.

---

## 🔧 Konfigurasi Sidebar

| Konfigurasi | Default | Keterangan |
|---|---|---|
| Penyedia | `Gemini` | Pilih `Gemini` (default) atau `Ollama`. |
| Nama model | `gemini-2.5-flash` | Bisa diganti ke `gemini-2.5-pro`, `llama3.1`, dst. |
| Temperature | `0.4` | Lebih tinggi → lebih kreatif. |
| Top K | `40` | Pembatas sampling. |
| Top P | `0.95` | Nucleus sampling. |
| Max Output Tokens | `1024` | Batas panjang jawaban. |
| Jumlah konteks RAG | `4` | Berapa banyak chunk diambil dari ChromaDB. |
| Riwayat percakapan | `5` | Berapa giliran dikirim ke model. |
| Gunakan Exa Web Search | `Off` | Aktif jika `EXA_API_KEY` tersedia. |

---

## 🗃️ Struktur Proyek

```
nusa-care-ai/
├── app.py                    # Entry point Streamlit
├── requirements.txt          # Dependensi Python
├── README.md                 # Dokumen ini
├── .env.example              # Contoh konfigurasi environment
├── .gitignore                # File yang diabaikan git
├── data/
│   └── knowledge_base.txt    # Sample basis pengetahuan NusaCare
├── src/
│   ├── __init__.py
│   ├── config.py             # Loader konfigurasi env
│   ├── gemini_client.py      # Klien LLM (Gemini / Ollama)
│   ├── rag.py                # Pipeline RAG + ChromaDB
│   ├── memory.py             # Memori percakapan per sesi
│   └── utils.py              # System prompt + helper Exa
└── chroma_db/                # Otomatis dibuat saat indeks dibangun
```

---

## 🧰 Tambahkan Dokumen ke Basis Pengetahuan

Untuk menambah informasi baru:

1. Tambahkan paragraf baru ke `data/knowledge_base.txt` (pisahkan setiap topik dengan baris kosong).
2. Klik **🔁 Bangun Ulang Indeks** di sidebar agar isi terbaru ter-embedding ulang.
3. Anda dapat mengganti seluruh isi file dengan dokumen lain (mis. FAQ instansi Anda) — pastikan format teks paragraf yang dipisah baris kosong.

> Catatan: setiap kali file `.txt` diubah, indeks ChromaDB harus dibangun ulang agar relevan.

---

## 🦙 Menggunakan Ollama (Opsional)

1. Pasang Ollama: <https://ollama.com/download>.
2. Tarik model yang diinginkan, misalnya:
   ```bash
   ollama pull llama3.1
   ```
3. Pastikan layanan Ollama berjalan (`ollama serve` jika belum otomatis).
4. Di sidebar NusaCare AI, pilih **Penyedia: Ollama** dan ganti **Nama model** menjadi nama model yang Anda tarik (mis. `llama3.1`).
5. Catatan: pembuatan indeks RAG **tetap memerlukan** `GEMINI_API_KEY` karena embedding menggunakan Gemini. Untuk inferensi murni lokal tanpa Gemini, Anda perlu mengganti embedder di `src/rag.py`.

---

## 🔐 Catatan Keamanan

- **Jangan pernah** meng-commit file `.env` ke repository.
- **Jangan** menaruh kunci API langsung di kode sumber.
- File `chroma_db/` sudah diabaikan oleh `.gitignore`; data embedding tetap di mesin Anda.
- Bot ini berbasis data fiktif untuk demo; jangan gunakan untuk operasi layanan publik nyata tanpa peninjauan keamanan dan privasi yang sesuai.

---

## 🛠️ Troubleshooting

**❌ "Kunci API Gemini tidak valid atau tidak diizinkan."**
- Periksa nilai `GEMINI_API_KEY` di file `.env`.
- Pastikan kunci aktif di Google AI Studio.
- Jalankan ulang aplikasi setelah mengubah `.env`.

**❌ "Kuota Gemini terlampaui."**
- Tunggu beberapa saat atau gunakan model yang lebih ringan (mis. `gemini-2.5-flash`).
- Periksa penggunaan di dashboard Google AI Studio.

**❌ "File basis pengetahuan tidak ditemukan."**
- Pastikan file `data/knowledge_base.txt` ada di lokasi yang benar.
- Periksa nilai `KNOWLEDGE_BASE_PATH` di `.env`.

**❌ "Tidak dapat terhubung ke Ollama."**
- Pastikan Ollama berjalan: `ollama list` harus menampilkan daftar model.
- Periksa `OLLAMA_BASE_URL` (default `http://localhost:11434`).

**🧹 Indeks "rusak" / pencarian aneh**
- Klik **🗑️ Reset Indeks** lalu **🔁 Bangun Ulang Indeks**.
- Atau hapus folder `chroma_db/` secara manual lalu jalankan ulang.

---

## 📄 Lisensi

Kode ini dibagikan untuk keperluan pembelajaran/demonstrasi. Sesuaikan lisensi pada repository Anda sesuai kebutuhan.

---

## 🤝 Kontribusi

Pull request dan saran perbaikan sangat dipersilakan. Pastikan perubahan tetap mempertahankan UI dan dokumentasi dalam Bahasa Indonesia, dan tidak menambahkan penyedia LLM selain Gemini atau Ollama.
