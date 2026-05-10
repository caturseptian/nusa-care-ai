"""Memori percakapan untuk satu sesi Streamlit.

Memori ini hanya bertahan selama browser tab aktif (in-session) — tidak
dipersistensi ke disk. Daftar pesan disimpan di luar (mis. di
`st.session_state`) agar tetap utuh di antara rerun Streamlit.
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict, List


class ChatMemory:
    """Pembungkus tipis di atas list dict pesan {role, content, ts}."""

    def __init__(self, store: List[Dict]):
        self._store = store

    def add_user(self, content: str) -> None:
        self._store.append(
            {
                "role": "user",
                "content": content,
                "ts": datetime.now().isoformat(timespec="seconds"),
            }
        )

    def add_assistant(self, content: str) -> None:
        self._store.append(
            {
                "role": "assistant",
                "content": content,
                "ts": datetime.now().isoformat(timespec="seconds"),
            }
        )

    def all(self) -> List[Dict]:
        return list(self._store)

    def recent(self, turns: int) -> List[Dict]:
        """Kembalikan paling banyak `turns` giliran terakhir (~2*turns pesan)."""
        if turns <= 0:
            return []
        return list(self._store[-(turns * 2):])

    def clear(self) -> None:
        self._store.clear()
