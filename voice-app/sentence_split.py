"""
Streaming sentence splitter.

Akkumuliert Token-Deltas in einem Puffer und gibt jeden vollständigen Satz
zurück, sobald er erkannt wird (.!?  gefolgt von Whitespace oder String-Ende).
Der Rest bleibt im Puffer und wird beim flush() am Stream-Ende ausgegeben.

Mindestlänge verhindert, dass kurze Fragmente ("Hm.", "Ja.") sofort an TTS
gehen – das gibt unnatürliche Audio-Stücke.
"""

from __future__ import annotations

import re
from typing import Iterator


# einfacher Splitter; reicht für Conversation-Sätze
_SENT_END = re.compile(r"([.!?…])(\s+|$)")
_MIN_LEN = 12  # Zeichen


class SentenceSplitter:
    def __init__(self, min_len: int = _MIN_LEN) -> None:
        self._buf = ""
        self._min_len = min_len

    def feed(self, delta: str) -> Iterator[str]:
        """Token-Delta einfügen, vollständige Sätze yielden."""
        if not delta:
            return
        self._buf += delta
        while True:
            m = _SENT_END.search(self._buf)
            if not m:
                return
            cut = m.end()
            sent = self._buf[:cut].strip()
            if len(sent) < self._min_len:
                # zu kurz – auf nächsten Satz warten
                return
            self._buf = self._buf[cut:]
            yield sent

    def flush(self) -> str:
        """Rest am Stream-Ende ausgeben (auch ohne Satzzeichen)."""
        rest = self._buf.strip()
        self._buf = ""
        return rest
