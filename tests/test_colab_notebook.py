import json
from pathlib import Path


def test_colab_notebook_is_one_click_and_uses_real_repo_and_full_pipeline():
    nb=json.loads(Path('notebooks/AMANAH_Training_Colab.ipynb').read_text(encoding='utf-8'))
    text='\n'.join(''.join(cell.get('source',[])) for cell in nb['cells'])
    assert 'YOUR_REPO_URL' not in text
    assert 'https://github.com/Saahar2001/Amanah.git' in text
    assert 'fetch_verified_sources.py' in text
    assert 'source_contract_smoke.py' in text
    assert 'qa_dataset.py' in text
    assert 'training.train' in text
    assert 'training.calibrate_thresholds' in text
    assert 'training.evaluate' in text
    assert 'package_hf_model.py' in text
    assert 'HF_TOKEN' in text
    assert 'create_repo' in text
    assert 'upload_folder' in text
    assert 'torch.cuda.is_available()' in text
    assert 'drive.mount' in text
    assert 'AMANAH_Artifacts' in text
    assert 'hf_model_repo_url' in text
