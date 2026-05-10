"""Utilitas: prompt sistem, formatting konteks, dan integrasi Exa opsional."""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """Anda adalah NusaCare AI, asisten layanan pelanggan resmi platform layanan publik digital NusaCare.

Karakter dan gaya bicara:
- Selalu menjawab dalam Bahasa Indonesia yang baik, sopan, ramah, dan profesional.
- Gunakan sapaan hangat seperti "Halo", "Selamat datang", atau "Tentu, dengan senang hati saya bantu" bila sesuai.
- Jangan menggurui. Jelaskan langkah-langkah secara ringkas, jelas, dan terstruktur (gunakan bullet/numbering bila perlu).

Aturan menjawab:
1. Prioritaskan informasi dari "KONTEKS BASIS PENGETAHUAN" yang diberikan. Jika konteks cukup, jawablah berdasarkan konteks tersebut dan boleh mengutip nomor sumbernya (mis. "Sumber 1").
2. Jika informasi tidak tersedia atau tidak cukup di basis pengetahuan, jawab dengan jujur:
   "Mohon maaf, informasi tersebut belum tersedia di basis pengetahuan kami. Silakan hubungi customer support NusaCare untuk bantuan lebih lanjut."
3. Jika pengguna menanyakan hal yang tidak berkaitan dengan layanan NusaCare (mis. trivia umum, opini politik, hiburan, perjudian), tolak dengan sopan dan arahkan kembali ke topik layanan NusaCare.
4. Jika pengguna meminta atau membagikan data sensitif (NIK lengkap, nomor kartu, kata sandi, OTP), tolak dengan sopan, jelaskan bahwa NusaCare tidak akan pernah meminta data tersebut, dan ingatkan agar mereka tidak membagikan data sensitif kepada siapa pun.
5. Jangan pernah mengarang nomor telepon, alamat kantor, jam operasional, biaya, atau prosedur. Jika ragu, gunakan kalimat di poin 2.
6. Jawablah dalam Bahasa Indonesia meskipun pengguna menulis dalam bahasa lain. Boleh menyertakan terjemahan singkat bila perlu.
7. Jika ada "KONTEKS WEB" tambahan, perlakukan sebagai informasi pendukung, bukan sumber utama; tetap utamakan basis pengetahuan resmi NusaCare.

Format jawaban:
- Mulai dengan jawaban langsung — hindari permohonan maaf yang panjang di awal.
- Gunakan poin atau langkah ber-nomor untuk prosedur.
- Akhiri dengan kalimat penutup singkat seperti "Semoga membantu" atau "Apakah ada hal lain yang ingin ditanyakan?" bila sesuai.
"""


def build_system_prompt() -> str:
    return SYSTEM_PROMPT


def format_context(chunks: List[Dict]) -> str:
    """Format chunk hasil retrieval menjadi blok konteks ber-label."""
    if not chunks:
        return "(tidak ada konteks yang ditemukan di basis pengetahuan)"
    parts: List[str] = []
    for i, ch in enumerate(chunks, start=1):
        text = (ch.get("document") or "").strip()
        meta = ch.get("metadata") or {}
        source = meta.get("source", "knowledge_base.txt")
        parts.append(f"[Sumber {i} - {source}]\n{text}")
    return "\n\n".join(parts)


def format_web_snippets(snippets: List[Dict]) -> str:
    if not snippets:
        return ""
    parts: List[str] = []
    for i, s in enumerate(snippets, start=1):
        title = s.get("title", "")
        url = s.get("url", "")
        text = (s.get("text") or "").strip()
        parts.append(f"[Web {i}] {title} ({url})\n{text}")
    return "\n\n".join(parts)


def exa_search(query: str, api_key: Optional[str], num_results: int = 3) -> List[Dict]:
    """Pencarian web via Exa. Mengembalikan [] jika gagal atau tidak dikonfigurasi."""
    if not api_key or not query.strip():
        return []
    try:
        from exa_py import Exa  # impor lokal agar pustaka bersifat opsional
    except ImportError:
        logger.warning("Pustaka exa_py tidak terpasang; melewati pencarian web.")
        return []
    try:
        client = Exa(api_key=api_key)
        result = client.search_and_contents(
            query,
            num_results=num_results,
            type="auto",
            text={"max_characters": 600},
        )
        items: List[Dict] = []
        for r in getattr(result, "results", []) or []:
            items.append(
                {
                    "title": getattr(r, "title", "") or "",
                    "url": getattr(r, "url", "") or "",
                    "text": getattr(r, "text", "") or "",
                }
            )
        return items
    except Exception as e:  # noqa: BLE001
        logger.warning("Pencarian Exa gagal: %s", e)
        return []


def compose_user_message(
    user_msg: str,
    context: str,
    web_snippets: str = "",
) -> str:
    """Susun pesan pengguna final yang dikirim ke LLM, mencakup blok konteks."""
    sections: List[str] = []
    sections.append("KONTEKS BASIS PENGETAHUAN:\n" + context)
    if web_snippets:
        sections.append("KONTEKS WEB (opsional, dari Exa):\n" + web_snippets)
    sections.append("PERTANYAAN PENGGUNA:\n" + user_msg.strip())
    return "\n\n".join(sections)
