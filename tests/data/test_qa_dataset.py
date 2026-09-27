import json
from pathlib import Path


def _write_jsonl(path: Path, rows: list[dict]):
    path.write_text('\n'.join(json.dumps(x, ensure_ascii=False) for x in rows)+'\n',encoding='utf-8')


def test_qa_dataset_reports_clean_reference_and_split_contract(tmp_path):
    from scripts.qa_dataset import qa_dataset
    bundle=tmp_path/'bundle.json'
    refs=[
        {'provider':'QuranEnc.com','translator':'A','version':'1','text':'A','source_url':'https://example.com/a','retrieved_at':'x','checksum':'sha256:a'},
        {'provider':'QuranEnc.com','translator':'B','version':'1','text':'B','source_url':'https://example.com/b','retrieved_at':'x','checksum':'sha256:b'},
        {'provider':'QuranEnc.com','translator':'C','version':'1','text':'C','source_url':'https://example.com/c','retrieved_at':'x','checksum':'sha256:c'},
    ]
    bundle.write_text(json.dumps({'ayahs':[{'ayah_id':'1:1','surah':1,'source_ar':'مصدر','references':refs}]},ensure_ascii=False),encoding='utf-8')
    split=tmp_path/'splits'; split.mkdir()
    row={'sample_id':'x','ayah_id':'1:1','surah':1,'source_type':'quran','source_ar':'مصدر','references_en':refs,'candidate_en':'A','labels':['FAITHFUL'],'severity':'S0','origin':'trusted_reference','metadata':{}}
    _write_jsonl(split/'train.jsonl',[row]); _write_jsonl(split/'validation.jsonl',[]); _write_jsonl(split/'test.jsonl',[])
    report=qa_dataset(bundle,split,expected_ayahs=1,expected_references=3,require_trainable_labels=False)
    assert report['status']=='ok'; assert report['reference_ayahs']==1; assert report['split_ayah_leakage']==[]; assert report['canonical_mismatches']==[]; assert report['label_counts']['FAITHFUL']==1


def test_qa_dataset_fails_on_ayah_leakage(tmp_path):
    from scripts.qa_dataset import qa_dataset
    bundle=tmp_path/'bundle.json'
    refs=[{'provider':'QuranEnc.com','translator':x,'version':'1','text':x,'source_url':'https://example.com','retrieved_at':'x','checksum':'sha256:x'} for x in ['A','B','C']]
    bundle.write_text(json.dumps({'ayahs':[{'ayah_id':'1:1','surah':1,'source_ar':'مصدر','references':refs}]},ensure_ascii=False),encoding='utf-8')
    split=tmp_path/'splits'; split.mkdir()
    row={'sample_id':'x','ayah_id':'1:1','surah':1,'source_type':'quran','source_ar':'مصدر','references_en':refs,'candidate_en':'A','labels':['FAITHFUL'],'severity':'S0','origin':'trusted_reference','metadata':{}}
    _write_jsonl(split/'train.jsonl',[row]); _write_jsonl(split/'validation.jsonl',[{**row,'sample_id':'y'}]); _write_jsonl(split/'test.jsonl',[])
    report=qa_dataset(bundle,split,expected_ayahs=1,expected_references=3,require_trainable_labels=False)
    assert report['status']=='failed'; assert report['split_ayah_leakage']==['1:1']
