import json
from pathlib import Path


def test_prepare_dataset_creates_faithful_and_mutated_group_splits(tmp_path:Path):
    from scripts.prepare_amanah_sd import prepare_dataset
    payload={"ayahs":[
        {"ayah_id":"1:1","surah":1,"source_ar":"مصدر 1","references":[{"provider":"QuranEnc","translator":"T","version":"1","text":"There is no compulsion in religion.","source_url":"https://quranenc.com/x","retrieved_at":"now","checksum":"sha256:x"}]},
        {"ayah_id":"1:2","surah":1,"source_ar":"مصدر 2","references":[{"provider":"QuranEnc","translator":"T","version":"1","text":"Some people may know, and they remember God.","source_url":"https://quranenc.com/x","retrieved_at":"now","checksum":"sha256:y"}]},
        {"ayah_id":"2:1","surah":2,"source_ar":"مصدر 3","references":[{"provider":"QuranEnc","translator":"T","version":"1","text":"If they repent, then forgive them.","source_url":"https://quranenc.com/x","retrieved_at":"now","checksum":"sha256:z"}]}
    ]}
    src=tmp_path/'refs.json'; src.write_text(json.dumps(payload),encoding='utf-8'); out=tmp_path/'out'; summary=prepare_dataset(src,out,seed=7); assert summary['total_samples']>3
    assert all(p.exists() for p in [out/'train.jsonl',out/'validation.jsonl',out/'test.jsonl',out/'dataset_summary.json'])
    seen={}
    for split in ('train','validation','test'):
        for line in (out/f'{split}.jsonl').read_text(encoding='utf-8').splitlines():
            row=json.loads(line); seen.setdefault(row['ayah_id'],set()).add(split)
    assert all(len(v)==1 for v in seen.values())
