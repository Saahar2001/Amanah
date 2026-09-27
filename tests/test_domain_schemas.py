import pytest
from pydantic import ValidationError


def test_taxonomy_and_request_contract():
    from data.models import DriftLabel, Severity, Decision
    from amanah_engine.schemas import AnalyzeRequest
    assert DriftLabel.NEGATION_FLIP.value == "NEGATION_FLIP"
    assert DriftLabel.SEMANTIC_GRADATION_LOSS.value == "SEMANTIC_GRADATION_LOSS"
    assert Severity.S3.value == "S3"
    assert Decision.ABSTAIN.value == "ABSTAIN"
    req = AnalyzeRequest(source_type="quran",source_ar="لَا إِكْرَاهَ فِي الدِّينِ",candidate_en="There is no compulsion in religion.",ayah_id="2:256")
    assert req.source_type == "quran"
    with pytest.raises(ValidationError): AnalyzeRequest(source_type="hadith", source_ar="x", candidate_en="y")


def test_reference_translation_is_frozen_and_requires_provenance():
    from data.models import ReferenceTranslation
    ref=ReferenceTranslation(provider="QuranEnc",translator="Example Translator",version="1.0.0",text="A faithful translation.",source_url="https://quranenc.com/example",retrieved_at="2026-09-27T00:00:00Z",checksum="sha256:abc123")
    with pytest.raises(ValidationError): ref.text = "changed"
    with pytest.raises(ValidationError): ReferenceTranslation(provider="QuranEnc",translator="Example Translator",version="",text="Text",source_url="",retrieved_at="2026-09-27T00:00:00Z",checksum="")


def test_sample_and_response_reject_unknown_labels():
    from data.models import AmanahSample, ReferenceTranslation
    from amanah_engine.schemas import AnalyzeResponse, DriftFinding
    ref=ReferenceTranslation(provider="QuranEnc",translator="Example Translator",version="1.0.0",text="A faithful translation.",source_url="https://quranenc.com/example",retrieved_at="2026-09-27T00:00:00Z",checksum="sha256:abc123")
    sample=AmanahSample(sample_id="amanah_1",ayah_id="1:1",surah=1,source_ar="بِسْمِ اللَّهِ",references_en=[ref],candidate_en="In the name of Allah",labels=["FAITHFUL"],severity="S0",origin="trusted_reference")
    assert sample.labels[0].value == "FAITHFUL"
    with pytest.raises(ValidationError): AmanahSample(sample_id="bad",ayah_id="1:1",surah=1,source_ar="x",references_en=[ref],candidate_en="y",labels=["MADE_UP"],severity="S0",origin="synthetic_mutation")
    finding=DriftFinding(label="NEGATION_FLIP",severity="S3",confidence=0.95)
    response=AnalyzeResponse(decision="CRITICAL",integrity_score=40,severity="S3",confidence=0.95,drifts=[finding],needs_human_review=True,model_version="amanah-smoke")
    assert response.drifts[0].label.value == "NEGATION_FLIP"
