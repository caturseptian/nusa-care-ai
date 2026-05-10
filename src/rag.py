"""RAG di atas ChromaDB dengan embedding Gemini.

Pipeline ringkas:
1. Baca file `knowledge_base.txt`.
2. Pecah teks menjadi chunk (paragraf + sliding window untuk paragraf panjang).
3. Embed setiap chunk menggunakan Gemini `text-embedding-004`.
4. Simpan ke ChromaDB persistent (`chroma_db/`).
5. Saat retrieval, embed query lalu lakukan similarity search.
"""
from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path
from typing import Callable, Dict, List, Optional

import chromadb
import google.generativeai as genai

logger = logging.getLogger(__name__)


def split_text(text: str, chunk_size: int = 500, overlap: int = 80) -> List[str]:
    """Pecah teks per paragraf; paragraf panjang dipotong dengan sliding window."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []
    for paragraph in paragraphs:
        if len(paragraph) <= chunk_size:
            chunks.append(paragraph)
            continue
        start = 0
        while start < len(paragraph):
            end = min(start + chunk_size, len(paragraph))
            chunks.append(paragraph[start:end])
            if end >= len(paragraph):
                break
            start = max(end - overlap, start + 1)
    return chunks


class GeminiEmbedder:
    """Embedder berbasis Gemini `text-embedding-004` (atau model embedding lain)."""

    def __init__(self, model: str = "gemini-embedding-001"):
        # `genai.embed_content` membutuhkan format "models/<name>".
        self.model = model if model.startswith("models/") else f"models/{model}"

    def _embed(self, text: str, task_type: str) -> List[float]:
        result = genai.embed_content(
            model=self.model,
            content=text,
            task_type=task_type,
        )
        emb = result.get("embedding") if isinstance(result, dict) else None
        if emb is None:
            raise RuntimeError("Embedding kosong dari Gemini.")
        # Beberapa versi SDK mengembalikan list-of-list ketika input adalah list.
        if isinstance(emb, list) and emb and isinstance(emb[0], list):
            emb = emb[0]
        return emb

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(t, "RETRIEVAL_DOCUMENT") for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text, "RETRIEVAL_QUERY")


class KnowledgeBase:
    """Pengelola koleksi ChromaDB untuk basis pengetahuan NusaCare."""

    def __init__(
        self,
        persist_dir: str,
        collection_name: str,
        embedder: GeminiEmbedder,
        kb_path: str,
    ):
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.embedder = embedder
        self.kb_path = kb_path
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self._get_or_create_collection()

    def _get_or_create_collection(self):
        return self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        try:
            return self.collection.count()
        except Exception:  # noqa: BLE001
            return 0

    def reset(self) -> None:
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:  # noqa: BLE001
            pass
        self.collection = self._get_or_create_collection()

    def build(
        self,
        force: bool = False,
        progress: Optional[Callable[[int, int, str], None]] = None,
    ) -> int:
        """Bangun indeks dari `kb_path`. Mengembalikan jumlah chunk yang diindeks."""
        if not os.path.exists(self.kb_path):
            raise FileNotFoundError(
                f"File basis pengetahuan tidak ditemukan: {self.kb_path}"
            )
        if force:
            self.reset()
        elif self.count() > 0:
            return self.count()

        with open(self.kb_path, "r", encoding="utf-8") as f:
            text = f.read()
        chunks = split_text(text)
        total = len(chunks)
        if total == 0:
            return 0

        ids: List[str] = []
        docs: List[str] = []
        embs: List[List[float]] = []
        metas: List[Dict] = []
        source_name = os.path.basename(self.kb_path)

        for i, chunk in enumerate(chunks):
            if progress:
                progress(i, total, "Membuat embedding dokumen")
            emb = self.embedder.embed_documents([chunk])[0]
            ids.append(f"chunk-{uuid.uuid4().hex[:12]}")
            docs.append(chunk)
            embs.append(emb)
            metas.append({"source": source_name, "chunk": i})

        self.collection.add(ids=ids, documents=docs, embeddings=embs, metadatas=metas)
        if progress:
            progress(total, total, "Selesai")
        return total

    def retrieve(self, query: str, k: int = 4) -> List[Dict]:
        if self.count() == 0:
            return []
        emb = self.embedder.embed_query(query)
        res = self.collection.query(query_embeddings=[emb], n_results=max(1, k))
        out: List[Dict] = []
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        for i, doc in enumerate(docs):
            out.append(
                {
                    "document": doc,
                    "metadata": metas[i] if i < len(metas) else {},
                    "distance": dists[i] if i < len(dists) else None,
                }
            )
        return out
