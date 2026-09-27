import json
from pathlib import Path


def test_training_notebook_uses_generic_repository_secret_and_full_pipeline():
    nb=json.loads(Path('notebooks/model_training_pipeline.ipynb').read_text(encoding='utf-8'))
    text='\n'.join(''.join(cell.get('source',[])) for cell in nb['cells'])
    assert 'REPOSITORY_URL' in text
    assert 'YOUR_REPO_URL' not in text
    assert 'fetch_verified_sources.py' in text
    assert 'source_contract_smoke.py' in text
    assert 'qa_dataset.py' in text
    assert 'training.train' in text
    assert 'training.calibrate_thresholds' in text
    assert 'training.evaluate' in text
    assert 'package_hf_model.py' in text
    assert 'HF_TOKEN' in text
    assert 'torch.cuda.is_available()' in text
    assert 'drive.mount' in text
    assert 'AMANAH_Artifacts' in text


def test_shared_files_do_not_embed_personal_names():
    blocked_terms=['sahar','mahmoud']
    paths=[
        Path('README.md'),
        Path('docs/PLATFORM_INTEGRATION_GUIDE.md'),
        Path('docs/PRESENTATION_REVISION_GUIDE.md'),
        Path('docs/QUALITY_ASSURANCE_CHECKLIST.md'),
        Path('docs/RELEASE_VALIDATION_STATUS.md'),
        Path('notebooks/model_training_pipeline.ipynb'),
    ]
    combined='\n'.join(p.read_text(encoding='utf-8').lower() for p in paths)
    for term in blocked_terms:
        assert term not in combined
