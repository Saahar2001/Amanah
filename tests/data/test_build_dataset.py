from copy import deepcopy


def make_ref():
    from data.models import ReferenceTranslation
    return ReferenceTranslation(provider="QuranEnc",translator="Example",version="1.0",text="There is no compulsion in religion.",source_url="https://quranenc.com/example",retrieved_at="2026-09-27T00:00:00Z",checksum="sha256:abc")


def test_build_faithful_samples_preserves_reference_bytes():
    from data.build_dataset import build_faithful_samples
    ref=make_ref(); before=ref.model_dump_json(); rows=build_faithful_samples(ayah_id="2:256",surah=2,source_ar="لَا إِكْرَاهَ فِي الدِّينِ",references=[ref])
    assert ref.model_dump_json()==before; assert len(rows)==1; assert rows[0].candidate_en==ref.text; assert [x.value for x in rows[0].labels]==["FAITHFUL"]; assert rows[0].severity.value=="S0"; assert rows[0].origin=="trusted_reference"; assert rows[0].metadata["reference_checksum"]=="sha256:abc"


def test_faithful_builder_requires_reference_metadata():
    from pydantic import ValidationError
    from data.models import ReferenceTranslation
    try:
        ReferenceTranslation(provider="QuranEnc",translator="Example",version="",text="x",source_url="https://quranenc.com/example",retrieved_at="now",checksum="")
    except ValidationError:
        return
    raise AssertionError("incomplete reference must be rejected")
