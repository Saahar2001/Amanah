import json
from pathlib import Path
import pytest


def test_registry_requires_complete_provenance(tmp_path: Path):
    from data.source_registry import load_source_registry, verify_source_record
    p=tmp_path/'registry.json'
    p.write_text(json.dumps([{'source_id':'quranenc-rowwad-en','provider':'QuranEnc','translator':'Rowwad Translation Center','language':'en','version':'1.0.19','source_url':'https://quranenc.com/en/browse/english_rwwad','retrieved_at':'2026-09-27T00:00:00Z','checksum':'sha256:abc','status':'approved_reference'}]),encoding='utf-8')
    records=load_source_registry(p); assert len(records)==1; assert verify_source_record(records[0]) is True
    with pytest.raises(ValueError,match='checksum'): verify_source_record(records[0].model_copy(update={'checksum':''}))


def test_registry_rejects_non_https_and_unknown_status(tmp_path: Path):
    from data.source_registry import load_source_registry
    p=tmp_path/'registry.json'; p.write_text(json.dumps([{'source_id':'x','provider':'x','translator':'x','language':'en','version':'1','source_url':'http://example.com','retrieved_at':'now','checksum':'sha256:x','status':'made_up'}]),encoding='utf-8')
    with pytest.raises(Exception): load_source_registry(p)


def test_reference_store_matches_normalized_arabic_and_refuses_ambiguous_match(tmp_path):
    from amanah_engine.references import JsonReferenceStore
    p=tmp_path/'refs.json'
    p.write_text(json.dumps({'ayahs':[{'ayah_id':'1:1','source_ar':'بِسْمِ اللَّهِ','references_en':['A'],'provenance_verified':True},{'ayah_id':'2:1','source_ar':'الم','references_en':['B'],'provenance_verified':True},{'ayah_id':'3:1','source_ar':'الٓمٓ','references_en':['C'],'provenance_verified':True}]},ensure_ascii=False),encoding='utf-8')
    store=JsonReferenceStore(p); assert store.match_source_ar('بسم الله')['ayah_id']=='1:1'; assert store.match_source_ar('الم') is None
