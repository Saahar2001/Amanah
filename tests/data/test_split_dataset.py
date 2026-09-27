from data.models import AmanahSample, DriftLabel, ReferenceTranslation, Severity


def make_sample(sample_id, ayah_id, surah):
    ref=ReferenceTranslation(provider='QuranEnc',translator='T',version='1',text='x',source_url='https://quranenc.com/x',retrieved_at='now',checksum='sha256:x')
    return AmanahSample(sample_id=sample_id,ayah_id=ayah_id,surah=surah,source_ar='a',references_en=[ref],candidate_en='x',labels=[DriftLabel.FAITHFUL],severity=Severity.S0,origin='trusted_reference')


def test_group_split_keeps_each_ayah_in_one_split():
    from data.split_dataset import group_split, assert_no_ayah_leakage
    samples=[make_sample('a1','1:1',1),make_sample('a2','1:1',1),make_sample('b1','1:2',1),make_sample('c1','2:1',2),make_sample('d1','3:1',3)]
    splits=group_split(samples,0.6,0.2,0.2,seed=7); assert_no_ayah_leakage(splits); locations={}
    for name,rows in splits.items():
        for row in rows: locations.setdefault(row.ayah_id,set()).add(name)
    assert all(len(v)==1 for v in locations.values())


def test_split_is_deterministic_for_seed():
    from data.split_dataset import group_split
    samples=[make_sample(str(i), f'1:{i}',1) for i in range(1,11)]
    a=group_split(samples,0.7,0.1,0.2,seed=42); b=group_split(samples,0.7,0.1,0.2,seed=42)
    assert [[x.sample_id for x in a[k]] for k in ('train','validation','test')] == [[x.sample_id for x in b[k]] for k in ('train','validation','test')]


def test_unseen_surah_holdout_removes_surah_from_training():
    from data.split_dataset import make_unseen_surah_holdout
    base,holdout=make_unseen_surah_holdout([make_sample('a','1:1',1),make_sample('b','2:1',2),make_sample('c','2:2',2)],{2})
    assert {x.surah for x in base}=={1}; assert {x.surah for x in holdout}=={2}
