from data.models import AmanahSample, DriftLabel, ReferenceTranslation, Severity


def sample(text="There is no compulsion in religion."):
    ref=ReferenceTranslation(provider="QuranEnc",translator="Example",version="1.0",text=text,source_url="https://quranenc.com/example",retrieved_at="2026-09-27T00:00:00Z",checksum="sha256:abc")
    return AmanahSample(sample_id="s1",ayah_id="2:256",surah=2,source_ar="لا إكراه في الدين",references_en=[ref],candidate_en=text,labels=[DriftLabel.FAITHFUL],severity=Severity.S0,origin="trusted_reference")


def test_validator_accepts_clean_single_mutation_and_rejects_unchanged_or_empty():
    from data.mutations import MutationResult
    from data.validators import validate_mutation
    base=sample()
    good=MutationResult(original=base.candidate_en,candidate="There is compulsion in religion.",label=DriftLabel.NEGATION_FLIP,changed_span="no",metadata={})
    assert validate_mutation(base,good).valid is True
    unchanged=MutationResult(original=base.candidate_en,candidate=base.candidate_en,label=DriftLabel.NEGATION_FLIP,changed_span=None,metadata={})
    assert validate_mutation(base,unchanged).valid is False
    empty=MutationResult(original=base.candidate_en,candidate="",label=DriftLabel.OMISSION,changed_span=None,metadata={})
    assert validate_mutation(base,empty).valid is False


def test_validator_rejects_reference_tampering_and_label_mismatch():
    from data.mutations import MutationResult
    from data.validators import validate_mutation
    base=sample(); tampered=base.model_copy(update={"source_ar":"غير الأصل"})
    result=MutationResult(original=base.candidate_en,candidate="There is compulsion in religion.",label=DriftLabel.NEGATION_FLIP,changed_span="no",metadata={})
    assert validate_mutation(tampered,result,expected_source_ar=base.source_ar).valid is False
    mismatch=MutationResult(original=base.candidate_en,candidate="All people know.",label=DriftLabel.NEGATION_FLIP,changed_span="Some",metadata={})
    assert validate_mutation(base,mismatch).valid is False


def test_validator_rejects_obvious_multi_error_corruption():
    from data.mutations import MutationResult
    from data.validators import validate_mutation
    base=sample("Some people may not know, and they remain silent.")
    result=MutationResult(original=base.candidate_en,candidate="All people must know.",label=DriftLabel.NEGATION_FLIP,changed_span="not",metadata={"operations":4})
    verdict=validate_mutation(base,result)
    assert verdict.valid is False
    assert "multiple" in verdict.reasons[0].lower()
