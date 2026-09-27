from data.models import AmanahSample, DriftLabel, ReferenceTranslation, Severity


def base_sample(text:str)->AmanahSample:
    ref=ReferenceTranslation(provider="QuranEnc",translator="Example",version="1.0",text=text,source_url="https://quranenc.com/example",retrieved_at="2026-09-27T00:00:00Z",checksum="sha256:abc")
    return AmanahSample(sample_id="s1",ayah_id="2:1",surah=2,source_ar="مصدر عربي",references_en=[ref],candidate_en=text,labels=[DriftLabel.FAITHFUL],severity=Severity.S0,origin="trusted_reference")

def test_negation_flip_only_when_eligible():
    from data.mutations import apply_negation_flip
    result=apply_negation_flip(base_sample("There is no compulsion in religion.")); assert result is not None; assert result.label==DriftLabel.NEGATION_FLIP; assert " no " not in f" {result.candidate.lower()} "; assert apply_negation_flip(base_sample("Mercy is near.")) is None

def test_quantifier_and_modality_mutations():
    from data.mutations import apply_quantifier_change, apply_modality_shift
    q=apply_quantifier_change(base_sample("Some people know.")); assert q is not None and "all" in q.candidate.lower() and q.label==DriftLabel.QUANTIFIER_CHANGE
    m=apply_modality_shift(base_sample("They may give charity.")); assert m is not None and "must" in m.candidate.lower() and m.label==DriftLabel.MODALITY_SHIFT; assert apply_modality_shift(base_sample("They give charity.")) is None

def test_omission_and_condition_loss_are_conservative():
    from data.mutations import apply_omission, apply_condition_loss
    o=apply_omission(base_sample("They pray at night, and they remember God.")); assert o is not None and len(o.candidate)<len(o.original)
    c=apply_condition_loss(base_sample("If they repent, then forgive them.")); assert c is not None and "if" not in c.candidate.lower(); assert apply_condition_loss(base_sample("Forgive them.")) is None
