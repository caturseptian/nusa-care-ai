"""NusaCare AI — Streamlit chatbot untuk platform layanan publik digital.

Jalankan dengan:
    streamlit run app.py
"""
from __future__ import annotations

import logging
from typing import Dict, List

import streamlit as st

from src.config import AppConfig, load_config, validate_gemini_key
from src.gemini_client import (
    GeminiBackend,
    GenerationParams,
    LLMClient,
    LLMError,
    OllamaBackend,
)
from src.memory import ChatMemory
from src.rag import GeminiEmbedder, KnowledgeBase
from src.utils import (
    build_system_prompt,
    compose_user_message,
    exa_search,
    format_context,
    format_web_snippets,
)

logging.basicConfig(level=logging.INFO)

st.set_page_config(
    page_title="NusaCare AI - Asisten Layanan Publik",
    page_icon="🇮🇩",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Inisialisasi session state
# ---------------------------------------------------------------------------
def init_state() -> None:
    if "history" not in st.session_state:
        st.session_state.history = []  # list of {role, content, ts}
    if "kb" not in st.session_state:
        st.session_state.kb = None
    if "kb_status" not in st.session_state:
        st.session_state.kb_status = "belum dibangun"
    if "flash" not in st.session_state:
        st.session_state.flash = None  # (kind, message) shown once after rerun


def render_flash() -> None:
    """Tampilkan pesan flash sekali, lalu hapus dari session state."""
    flash = st.session_state.get("flash")
    if not flash:
        return
    kind, msg = flash
    {
        "success": st.success,
        "error": st.error,
        "info": st.info,
        "warning": st.warning,
    }.get(kind, st.info)(msg)
    st.session_state.flash = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_kb(cfg: AppConfig) -> KnowledgeBase:
    """Inisialisasi (lazily) `KnowledgeBase` dan simpan di session state."""
    if st.session_state.kb is None:
        embedder = GeminiEmbedder(model=cfg.gemini_embedding_model)
        st.session_state.kb = KnowledgeBase(
            persist_dir=cfg.chroma_persist_dir,
            collection_name=cfg.chroma_collection,
            embedder=embedder,
            kb_path=cfg.knowledge_base_path,
        )
        cnt = st.session_state.kb.count()
        st.session_state.kb_status = (
            f"siap ({cnt} chunk)" if cnt > 0 else "belum dibangun"
        )
    return st.session_state.kb


def build_llm_client(provider: str, model_name: str, cfg: AppConfig) -> LLMClient:
    if provider == "Gemini":
        if not validate_gemini_key(cfg.gemini_api_key):
            raise LLMError("Kunci API Gemini belum dikonfigurasi pada file .env.")
        backend = GeminiBackend(api_key=cfg.gemini_api_key, model=model_name)
    else:
        backend = OllamaBackend(base_url=cfg.ollama_base_url, model=model_name)
    return LLMClient(backend)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar(cfg: AppConfig) -> Dict:
    with st.sidebar:
        st.title("⚙️ Konfigurasi NusaCare AI")
        st.caption(
            "Atur penyedia model, parameter generasi, dan basis pengetahuan."
        )

        st.subheader("Penyedia Model")
        provider = st.radio(
            "Pilih penyedia",
            options=["Gemini", "Ollama"],
            index=0,
            help="Gemini adalah default. Ollama bersifat opsional dan memerlukan instalasi lokal.",
        )

        default_model = cfg.gemini_model if provider == "Gemini" else cfg.ollama_model
        model_name = st.text_input("Nama model", value=default_model)

        st.subheader("Parameter Generasi")
        temperature = st.slider(
            "Temperature", 0.0, 2.0, cfg.generation.temperature, 0.05,
            help="Lebih tinggi = lebih kreatif. Untuk customer service, 0.2-0.6 disarankan.",
        )
        top_k = st.slider("Top K", 1, 100, cfg.generation.top_k, 1)
        top_p = st.slider("Top P", 0.0, 1.0, cfg.generation.top_p, 0.05)
        max_output_tokens = st.slider(
            "Max Output Tokens", 128, 8192, cfg.generation.max_output_tokens, 64
        )

        st.subheader("RAG & Memori")
        rag_k = st.slider(
            "Jumlah konteks RAG (top-k)", 1, 10, 4, 1,
            help="Berapa banyak potongan basis pengetahuan yang diambil per pertanyaan.",
        )
        history_turns = st.slider(
            "Riwayat percakapan (turn)", 1, 20, 5, 1,
            help="Berapa banyak giliran percakapan terakhir yang dikirim ke model.",
        )

        st.subheader("Pencarian Web (Opsional)")
        exa_available = bool(cfg.exa_api_key)
        if not exa_available:
            st.caption(
                "ℹ️ Setel `EXA_API_KEY` di file .env untuk mengaktifkan Exa Web Search."
            )
        use_exa = st.toggle(
            "Gunakan Exa Web Search",
            value=False,
            disabled=not exa_available,
            help="Hanya tersedia bila EXA_API_KEY dikonfigurasi.",
        )

        st.subheader("Aksi")
        col1, col2 = st.columns(2)
        with col1:
            rebuild = st.button(
                "🔁 Bangun Ulang Indeks", use_container_width=True
            )
        with col2:
            reset = st.button("🗑️ Reset Indeks", use_container_width=True)
        clear = st.button("🧹 Bersihkan Riwayat", use_container_width=True)

        st.subheader("Status")
        api_ok = validate_gemini_key(cfg.gemini_api_key)
        st.write(
            "✅ Gemini API: siap" if api_ok else "❌ Gemini API: belum dikonfigurasi"
        )
        st.write(
            "✅ Exa API: siap" if exa_available else "⚪ Exa API: tidak diatur"
        )
        st.write(
            f"📚 Indeks: {st.session_state.get('kb_status', 'belum dibangun')}"
        )

        with st.expander("Tentang NusaCare AI"):
            st.markdown(
                "**NusaCare AI** adalah asisten layanan pelanggan untuk platform "
                "layanan publik digital fiktif **NusaCare**. Aplikasi ini menggunakan "
                "**Google Gemini** sebagai LLM utama, **ChromaDB** untuk RAG, dan "
                "**Streamlit** sebagai antarmuka. Kode bersifat sumber terbuka untuk "
                "tujuan pembelajaran."
            )

    return {
        "provider": provider,
        "model_name": model_name.strip() or default_model,
        "params": GenerationParams(
            temperature=float(temperature),
            top_k=int(top_k),
            top_p=float(top_p),
            max_output_tokens=int(max_output_tokens),
        ),
        "rag_k": int(rag_k),
        "history_turns": int(history_turns),
        "use_exa": bool(use_exa),
        "rebuild": bool(rebuild),
        "reset": bool(reset),
        "clear": bool(clear),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    init_state()
    cfg = load_config()

    st.title("🇮🇩 NusaCare AI")
    st.caption(
        "Asisten layanan pelanggan untuk platform layanan publik digital NusaCare. "
        "Tanyakan tentang prosedur, persyaratan, dan FAQ layanan kami."
    )

    settings = render_sidebar(cfg)
    render_flash()
    memory = ChatMemory(st.session_state.history)

    # --- Validasi konfigurasi -------------------------------------------------
    if settings["provider"] == "Gemini" and not validate_gemini_key(cfg.gemini_api_key):
        st.error(
            "🔑 **Kunci API Gemini belum dikonfigurasi.**\n\n"
            "Langkah-langkah:\n"
            "1. Salin `.env.example` menjadi `.env`.\n"
            "2. Dapatkan kunci API di https://aistudio.google.com/app/apikey.\n"
            "3. Isi `GEMINI_API_KEY=...` pada file `.env`.\n"
            "4. Jalankan ulang aplikasi.\n\n"
            "Atau pilih penyedia **Ollama** di sidebar untuk inferensi lokal."
        )
        st.stop()

    # --- Inisialisasi knowledge base -----------------------------------------
    kb = None
    if validate_gemini_key(cfg.gemini_api_key):
        try:
            kb = get_kb(cfg)
        except Exception as e:  # noqa: BLE001
            st.warning(f"Gagal menginisialisasi ChromaDB: {e}")
    else:
        st.info(
            "ℹ️ Embedding Gemini memerlukan `GEMINI_API_KEY`. Indeks RAG dinonaktifkan "
            "selama berjalan dengan Ollama tanpa kunci Gemini."
        )

    # --- Aksi sidebar ---------------------------------------------------------
    if settings["clear"]:
        memory.clear()
        st.session_state.flash = ("success", "Riwayat percakapan telah dibersihkan.")
        st.rerun()

    if kb is not None and settings["reset"]:
        try:
            kb.reset()
            st.session_state.kb_status = "belum dibangun"
            st.session_state.flash = ("success", "Indeks ChromaDB telah direset.")
            st.rerun()
        except Exception as e:  # noqa: BLE001
            st.error(f"Gagal mereset indeks: {e}")

    if kb is not None and settings["rebuild"]:
        try:
            progress_bar = st.progress(0, text="Membangun indeks...")

            def _on_progress(i: int, total: int, msg: str) -> None:
                pct = int((i / max(1, total)) * 100)
                progress_bar.progress(
                    min(100, pct), text=f"{msg} ({i}/{total})"
                )

            count = kb.build(force=True, progress=_on_progress)
            progress_bar.empty()
            st.session_state.kb_status = f"siap ({count} chunk)"
            st.session_state.flash = (
                "success",
                f"Indeks berhasil dibangun: {count} chunk telah diindeks.",
            )
            st.rerun()
        except FileNotFoundError as e:
            st.error(str(e))
        except Exception as e:  # noqa: BLE001
            st.error(f"Gagal membangun indeks: {e}")

    if kb is not None and kb.count() == 0:
        st.info(
            "ℹ️ Indeks ChromaDB masih kosong. Klik **🔁 Bangun Ulang Indeks** "
            "di sidebar untuk mengindeks file `data/knowledge_base.txt`."
        )

    # --- Render riwayat chat --------------------------------------------------
    for msg in memory.all():
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # --- Input baru -----------------------------------------------------------
    user_msg = st.chat_input(
        "Tulis pertanyaan Anda di sini... "
        "(mis. 'Bagaimana cara mendaftar akun NusaCare?')"
    )
    if not user_msg:
        return

    memory.add_user(user_msg)
    with st.chat_message("user"):
        st.markdown(user_msg)

    # --- Retrieval RAG --------------------------------------------------------
    chunks: List[Dict] = []
    if kb is not None and kb.count() > 0:
        try:
            chunks = kb.retrieve(user_msg, k=settings["rag_k"])
        except Exception as e:  # noqa: BLE001
            st.warning(f"Pengambilan konteks gagal: {e}")

    context_text = format_context(chunks)

    web_text = ""
    if settings["use_exa"] and cfg.exa_api_key:
        with st.spinner("Mencari informasi tambahan via Exa..."):
            snippets = exa_search(user_msg, cfg.exa_api_key, num_results=3)
        web_text = format_web_snippets(snippets)

    final_user_msg = compose_user_message(user_msg, context_text, web_text)
    system_prompt = build_system_prompt()

    # Riwayat sebelumnya (tanpa pesan user yang baru saja ditambahkan).
    prior = memory.recent(settings["history_turns"])
    if prior and prior[-1]["role"] == "user":
        prior = prior[:-1]
    history_for_llm = [
        {"role": m["role"], "content": m["content"]} for m in prior
    ]

    # --- Bangun klien LLM -----------------------------------------------------
    try:
        client = build_llm_client(
            settings["provider"], settings["model_name"], cfg
        )
    except LLMError as e:
        st.error(str(e))
        return

    # --- Streaming jawaban ----------------------------------------------------
    with st.chat_message("assistant"):
        placeholder = st.empty()
        accumulated = ""
        try:
            for chunk in client.stream(
                system_prompt=system_prompt,
                history=history_for_llm,
                user_msg=final_user_msg,
                params=settings["params"],
            ):
                accumulated += chunk
                placeholder.markdown(accumulated + "▌")
            placeholder.markdown(accumulated or "_(jawaban kosong)_")
        except LLMError as e:
            placeholder.markdown(f"⚠️ {e}")
            accumulated = f"⚠️ {e}"
        except Exception as e:  # noqa: BLE001
            placeholder.markdown(f"⚠️ Terjadi galat tak terduga: {e}")
            accumulated = f"⚠️ Terjadi galat tak terduga: {e}"

    if accumulated:
        memory.add_assistant(accumulated)


if __name__ == "__main__":
    main()
