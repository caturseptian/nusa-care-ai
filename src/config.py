"""Konfigurasi aplikasi NusaCare AI.

Memuat variabel environment dari `.env` (jika ada) lalu mengekspos sebagai
dataclass `AppConfig` yang mudah dipakai dari kode lain.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


@dataclass
class GenerationParams:
    """Parameter generasi default untuk LLM."""

    temperature: float = 0.4
    top_k: int = 40
    top_p: float = 0.95
    max_output_tokens: int = 1024


@dataclass
class AppConfig:
    gemini_api_key: Optional[str]
    gemini_model: str
    gemini_embedding_model: str
    ollama_base_url: str
    ollama_model: str
    exa_api_key: Optional[str]
    chroma_persist_dir: str
    chroma_collection: str
    knowledge_base_path: str
    generation: GenerationParams = field(default_factory=GenerationParams)


def _get(name: str, default: Optional[str] = None) -> Optional[str]:
    val = os.getenv(name)
    if val is None:
        return default
    val = val.strip()
    if val == "" or val.lower() == "your_gemini_api_key_here":
        return default
    return val


def load_config() -> AppConfig:
    """Membaca konfigurasi dari environment dan mengembalikan AppConfig."""
    return AppConfig(
        gemini_api_key=_get("GEMINI_API_KEY"),
        gemini_model=_get("GEMINI_MODEL", "gemini-2.5-flash") or "gemini-2.5-flash",
        gemini_embedding_model=_get("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
        or "gemini-embedding-001",
        ollama_base_url=_get("OLLAMA_BASE_URL", "http://localhost:11434")
        or "http://localhost:11434",
        ollama_model=_get("OLLAMA_MODEL", "llama3.1") or "llama3.1",
        exa_api_key=_get("EXA_API_KEY"),
        chroma_persist_dir=_get("CHROMA_PERSIST_DIR", "./chroma_db") or "./chroma_db",
        chroma_collection=_get("CHROMA_COLLECTION", "nusacare_kb") or "nusacare_kb",
        knowledge_base_path=_get("KNOWLEDGE_BASE_PATH", "./data/knowledge_base.txt")
        or "./data/knowledge_base.txt",
    )


def validate_gemini_key(key: Optional[str]) -> bool:
    """Pemeriksaan format ringan. Validasi sebenarnya terjadi pada panggilan API pertama."""
    if not key:
        return False
    key = key.strip()
    if len(key) < 20:
        return False
    if key.lower() == "your_gemini_api_key_here":
        return False
    return True
