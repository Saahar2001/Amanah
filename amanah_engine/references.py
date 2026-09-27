from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

_ARABIC_MARKS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED\u0640]")

def normalize_arabic(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text))
    text = _ARABIC_MARKS.sub("", text)
    text = text.replace("ٱ", "ا").replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    return " ".join(text.split()).strip()


class JsonReferenceStore:
    def __init__(self, path: str | Path):
        self.path=Path(path); self._by_ayah={}; self._by_source_ar={}
        if self.path.exists():
            payload=json.loads(self.path.read_text(encoding='utf-8'))
            rows=payload.get('ayahs',payload) if isinstance(payload,dict) else payload
            self._by_ayah={row['ayah_id']:row for row in rows}
            for row in rows:
                source = row.get('source_ar')
                if source:
                    key = normalize_arabic(source)
                    # Keep only unique canonical matches. Duplicate normalized text is unsafe to auto-resolve.
                    if key in self._by_source_ar:
                        self._by_source_ar[key] = None
                    else:
                        self._by_source_ar[key] = row

    def lookup(self, ayah_id: str | None):
        if not ayah_id: return None
        return self._by_ayah.get(ayah_id)

    def match_source_ar(self, source_ar: str | None):
        if not source_ar: return None
        return self._by_source_ar.get(normalize_arabic(source_ar))

    def __len__(self) -> int:
        return len(self._by_ayah)
