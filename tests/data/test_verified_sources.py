def test_parse_tanzil_txt_2_extracts_ayah_ids():
    from scripts.fetch_verified_sources import parse_tanzil_txt2
    payload="# header\n1|1|بِسْمِ اللَّهِ\n1|2|الْحَمْدُ لِلَّهِ\n"
    assert parse_tanzil_txt2(payload)=={"1:1":"بِسْمِ اللَّهِ","1:2":"الْحَمْدُ لِلَّهِ"}


def test_quranenc_list_selection_requires_three_approved_keys():
    from scripts.fetch_verified_sources import select_english_translations
    payload=[{"key":"english_rwwad","language_iso_code":"en","version":"1.0.19","title":"Rowwad"},{"key":"english_saheeh","language_iso_code":"en","version":"1.1.2","title":"Noor"},{"key":"english_hilali_khan","language_iso_code":"en","version":"1.1.2","title":"Hilali"},{"key":"english_waleed","language_iso_code":"en","version":"1.0.2","title":"In progress"}]
    assert [x["key"] for x in select_english_translations(payload)]==["english_rwwad","english_saheeh","english_hilali_khan"]


def test_build_reference_bundle_preserves_canonical_and_provenance():
    from scripts.fetch_verified_sources import build_reference_bundle
    arabic={"1:1":"نص عربي"}
    metadata=[{"key":"english_rwwad","version":"1.0.19","title":"Rowwad Translation Center"},{"key":"english_saheeh","version":"1.1.2","title":"Noor International Center"},{"key":"english_hilali_khan","version":"1.1.2","title":"Hilali and Khan"}]
    translations={"english_rwwad":{"1:1":"Reference A"},"english_saheeh":{"1:1":"Reference B"},"english_hilali_khan":{"1:1":"Reference C"}}
    bundle,store=build_reference_bundle(arabic,metadata,translations,retrieved_at="2026-09-27T00:00:00Z")
    row=bundle["ayahs"][0]
    assert row["source_ar"]=="نص عربي"; assert len(row["references"])==3; assert all(r["checksum"].startswith("sha256:") for r in row["references"])
    assert store["ayahs"][0]["provenance_verified"] is True
    assert store["ayahs"][0]["references_en"]==["Reference A","Reference B","Reference C"]


def test_parse_quranenc_sura_accepts_list_and_result_wrapper():
    from scripts.fetch_verified_sources import parse_quranenc_sura
    assert parse_quranenc_sura([{"sura":"1","aya":"1","translation":"A"}])=={"1:1":"A"}
    assert parse_quranenc_sura({"result":[{"sura":1,"aya":2,"translation":"B"}]})=={"1:2":"B"}
