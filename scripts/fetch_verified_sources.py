from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

APPROVED_ENGLISH_KEYS = (
    "english_rwwad",
    "english_saheeh",
    "english_hilali_khan",
)
QURANENC_LIST_URL = "https://quranenc.com/api/v1/translations/list/en/?localization=en"
QURANENC_SURA_URL = "https://quranenc.com/api/v1/translation/sura/{key}/{sura}"
TANZIL_DOWNLOAD_URL = "https://tanzil.net/pub/download/index.php"
TANZIL_FORM = {
    "quranType": "uthmani",
    "outType": "txt-2",
    "marks": "true",
    "sajdah": "true",
    "tatweel": "true",
    "agree": "true",
}
EXPECTED_AYAH_COUNT = 6236
TRANSLATOR_NAMES = {
    "english_rwwad": "Rowwad Translation Center",
    "english_saheeh": "Noor International Center",
    "english_hilali_khan": "Taqi-ud-Din al-Hilali and Muhammad Muhsin Khan",
}


def _sha256_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _unwrap_rows(payload: Any) -> list[dict]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("result", "translations", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
    raise ValueError("Unsupported QuranEnc response structure")


def parse_tanzil_txt2(text: str) -> dict[str, str]:
    rows: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip("\ufeff\n\r")
        if not line or line.startswith("#"):
            continue
        parts = line.split("|", 2)
        if len(parts) != 3:
            continue
        sura, ayah, arabic = parts
        if not (sura.isdigit() and ayah.isdigit() and arabic.strip()):
            continue
        rows[f"{int(sura)}:{int(ayah)}"] = arabic.strip()
    return rows


def select_english_translations(payload: Any) -> list[dict]:
    rows = _unwrap_rows(payload)
    by_key = {str(row.get("key")): row for row in rows}
    missing = [key for key in APPROVED_ENGLISH_KEYS if key not in by_key]
    if missing:
        raise ValueError(f"Required QuranEnc translations missing: {', '.join(missing)}")
    return [by_key[key] for key in APPROVED_ENGLISH_KEYS]


def parse_quranenc_sura(payload: Any) -> dict[str, str]:
    rows = _unwrap_rows(payload)
    out: dict[str, str] = {}
    for row in rows:
        sura = int(row["sura"])
        ayah = int(row["aya"])
        text = str(row["translation"]).strip()
        if not text:
            raise ValueError(f"Empty QuranEnc translation at {sura}:{ayah}")
        out[f"{sura}:{ayah}"] = text
    return out


def build_reference_bundle(arabic: dict[str, str], translation_metadata: list[dict], translations: dict[str, dict[str, str]], *, retrieved_at: str) -> tuple[dict, dict]:
    keys = [str(row["key"]) for row in translation_metadata]
    if tuple(keys) != APPROVED_ENGLISH_KEYS:
        raise ValueError("Translation metadata order must match approved English reference set")
    ayahs = []
    runtime_rows = []
    for ayah_id in sorted(arabic, key=lambda x: tuple(map(int, x.split(":")))):
        surah = int(ayah_id.split(":", 1)[0])
        refs = []
        runtime_refs = []
        for meta in translation_metadata:
            key = str(meta["key"])
            if ayah_id not in translations.get(key, {}):
                raise ValueError(f"Missing {key} translation for {ayah_id}")
            translated = translations[key][ayah_id]
            refs.append({
                "provider": "QuranEnc.com",
                "translator": TRANSLATOR_NAMES.get(key, str(meta.get("title") or key)),
                "version": str(meta.get("version") or "unknown"),
                "text": translated,
                "source_url": f"https://quranenc.com/api/v1/translation/aya/{key}/{ayah_id.replace(':', '/')}",
                "retrieved_at": retrieved_at,
                "checksum": _sha256_text(translated),
            })
            runtime_refs.append(translated)
        ayahs.append({"ayah_id": ayah_id,"surah": surah,"source_ar": arabic[ayah_id],"references": refs})
        runtime_rows.append({"ayah_id": ayah_id,"surah": surah,"source_ar": arabic[ayah_id],"references_en": runtime_refs,"provenance_verified": True,"canonical_source": "Tanzil Quran Text — Uthmani v1.1","canonical_source_url": "https://tanzil.net/download/","reference_provider": "QuranEnc.com","reference_keys": keys,"reference_versions": [str(meta.get("version") or "unknown") for meta in translation_metadata],"retrieved_at": retrieved_at})
    bundle = {"metadata": {"canonical_source": "Tanzil Quran Text — Uthmani v1.1","canonical_license": "CC BY 3.0; verbatim text only","canonical_url": "https://tanzil.net/download/","reference_provider": "QuranEnc.com","reference_keys": keys,"retrieved_at": retrieved_at},"ayahs": ayahs}
    store = {"metadata": bundle["metadata"], "ayahs": runtime_rows}
    return bundle, store


def _build_http_session():
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (compatible; SemanticIntegrityResearch/0.1)",
        "Accept": "application/json,text/plain,*/*",
    })
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        status=5,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET", "POST"}),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _fetch_tanzil_text(session, timeout: int) -> str:
    errors = []
    for method in ("get", "post"):
        try:
            if method == "get":
                response = session.get(TANZIL_DOWNLOAD_URL, params=TANZIL_FORM, timeout=timeout)
            else:
                response = session.post(TANZIL_DOWNLOAD_URL, data=TANZIL_FORM, timeout=timeout)
            response.raise_for_status()
            parsed = parse_tanzil_txt2(response.text)
            if len(parsed) == EXPECTED_AYAH_COUNT:
                return response.text
            errors.append(f"{method.upper()} returned {len(parsed)} parsed ayahs")
        except Exception as exc:
            errors.append(f"{method.upper()}: {type(exc).__name__}: {exc}")
    raise RuntimeError("Unable to retrieve verified Tanzil text. " + " | ".join(errors))


def fetch_verified_sources(*, timeout: int = 60, session=None) -> tuple[dict, dict]:
    if session is None:
        session = _build_http_session()
    retrieved_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    tanzil_text = _fetch_tanzil_text(session, timeout)
    arabic = parse_tanzil_txt2(tanzil_text)
    if len(arabic) != EXPECTED_AYAH_COUNT:
        raise ValueError(f"Expected {EXPECTED_AYAH_COUNT} Tanzil ayahs, got {len(arabic)}")
    listing = session.get(QURANENC_LIST_URL, timeout=timeout); listing.raise_for_status()
    metadata = select_english_translations(listing.json())
    translations: dict[str, dict[str, str]] = {key: {} for key in APPROVED_ENGLISH_KEYS}
    for key in APPROVED_ENGLISH_KEYS:
        for sura in range(1, 115):
            response = session.get(QURANENC_SURA_URL.format(key=key, sura=sura), timeout=timeout); response.raise_for_status()
            translations[key].update(parse_quranenc_sura(response.json()))
        if len(translations[key]) != EXPECTED_AYAH_COUNT:
            raise ValueError(f"Expected {EXPECTED_AYAH_COUNT} ayahs for {key}, got {len(translations[key])}")
    return build_reference_bundle(arabic, metadata, translations, retrieved_at=retrieved_at)


def main() -> None:
    p = argparse.ArgumentParser(description="Fetch verified AMANAH Qur'an Arabic + English reference bundle")
    p.add_argument("--output", default="data/private/reference_bundle.json")
    p.add_argument("--runtime-store", default="data/reference_store.json")
    p.add_argument("--timeout", type=int, default=60)
    args = p.parse_args()
    bundle, store = fetch_verified_sources(timeout=args.timeout)
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    runtime = Path(args.runtime_store); runtime.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
    runtime.write_text(json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"ayahs": len(bundle["ayahs"]),"references_per_ayah": len(bundle["ayahs"][0]["references"]) if bundle["ayahs"] else 0,"output": str(output),"runtime_store": str(runtime)}, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
