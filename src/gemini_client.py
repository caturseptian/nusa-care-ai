"""Klien LLM terpadu yang mendukung Gemini (default) dan Ollama (opsional).

Antarmuka utama: `LLMClient.stream(...)` yang menghasilkan iterator string token.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Dict, Iterator, List

import google.generativeai as genai
import requests

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Galat penyedia LLM yang sudah dipetakan ke pesan Bahasa Indonesia."""


@dataclass
class GenerationParams:
    temperature: float = 0.4
    top_k: int = 40
    top_p: float = 0.95
    max_output_tokens: int = 1024


class GeminiBackend:
    def __init__(self, api_key: str, model: str):
        if not api_key:
            raise LLMError("Kunci API Gemini belum dikonfigurasi.")
        genai.configure(api_key=api_key)
        self.model_name = model

    def stream(
        self,
        system_prompt: str,
        history: List[Dict],
        user_msg: str,
        params: GenerationParams,
    ) -> Iterator[str]:
        try:
            model = genai.GenerativeModel(
                model_name=self.model_name,
                system_instruction=system_prompt,
            )
            gemini_history: List[Dict] = []
            for m in history:
                role = "user" if m["role"] == "user" else "model"
                gemini_history.append({"role": role, "parts": [m["content"]]})
            chat = model.start_chat(history=gemini_history)
            response = chat.send_message(
                user_msg,
                generation_config={
                    "temperature": params.temperature,
                    "top_k": params.top_k,
                    "top_p": params.top_p,
                    "max_output_tokens": params.max_output_tokens,
                },
                stream=True,
            )
            for chunk in response:
                text = getattr(chunk, "text", None)
                if text:
                    yield text
        except LLMError:
            raise
        except Exception as e:  # noqa: BLE001
            msg = str(e).lower()
            if (
                "api key" in msg
                or "permission" in msg
                or "unauthorized" in msg
                or "401" in msg
                or "invalid" in msg and "key" in msg
            ):
                raise LLMError(
                    "Kunci API Gemini tidak valid atau tidak diizinkan. "
                    "Periksa GEMINI_API_KEY pada file .env."
                ) from e
            if "quota" in msg or "rate" in msg or "429" in msg:
                raise LLMError(
                    "Kuota Gemini terlampaui. Coba lagi nanti atau gunakan model lain."
                ) from e
            if "not found" in msg or "404" in msg:
                raise LLMError(
                    f"Model Gemini '{self.model_name}' tidak ditemukan. "
                    "Pastikan nama model benar."
                ) from e
            raise LLMError(f"Terjadi galat saat memanggil Gemini: {e}") from e


class OllamaBackend:
    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.model_name = model

    def stream(
        self,
        system_prompt: str,
        history: List[Dict],
        user_msg: str,
        params: GenerationParams,
    ) -> Iterator[str]:
        messages: List[Dict] = [{"role": "system", "content": system_prompt}]
        for m in history:
            messages.append({"role": m["role"], "content": m["content"]})
        messages.append({"role": "user", "content": user_msg})

        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": params.temperature,
                "top_k": params.top_k,
                "top_p": params.top_p,
                "num_predict": params.max_output_tokens,
            },
        }
        try:
            with requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                stream=True,
                timeout=300,
            ) as r:
                if r.status_code != 200:
                    raise LLMError(
                        f"Ollama mengembalikan status {r.status_code}: {r.text[:200]}"
                    )
                for line in r.iter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line.decode("utf-8"))
                    except json.JSONDecodeError:
                        continue
                    msg = data.get("message", {})
                    content = msg.get("content")
                    if content:
                        yield content
                    if data.get("done"):
                        break
        except requests.ConnectionError as e:
            raise LLMError(
                f"Tidak dapat terhubung ke Ollama di {self.base_url}. "
                "Pastikan layanan Ollama sedang berjalan."
            ) from e
        except LLMError:
            raise
        except Exception as e:  # noqa: BLE001
            raise LLMError(f"Terjadi galat saat memanggil Ollama: {e}") from e


class LLMClient:
    """Klien chat streaming yang agnostik terhadap penyedia."""

    def __init__(self, backend):
        self.backend = backend

    def stream(
        self,
        system_prompt: str,
        history: List[Dict],
        user_msg: str,
        params: GenerationParams,
    ) -> Iterator[str]:
        yield from self.backend.stream(system_prompt, history, user_msg, params)
