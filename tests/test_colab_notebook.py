import json
from pathlib import Path


def test_training_notebook_runs_clean_pipeline():
    path = Path("notebooks/model_training.ipynb")
    nb = json.loads(path.read_text(encoding="utf-8"))
    text = "\n".join("".join(cell.get("source", [])) for cell in nb["cells"])

    assert "REPOSITORY_ID = 1391259908" in text
    assert "NOTEBOOK_RELEASE = \"2026.09.28-r6\"" in text
    assert "/zipball/" in text
    assert "git\", \"clone\"" not in text
    assert "YOUR_REPO_URL" not in text
    assert "--upgrade" not in text
    assert "setuptools" not in text
    assert "wheel" not in text
    assert "pip\", \"install\", \"-q\", \"-r\", \"requirements.txt" not in text
    assert "fetch_verified_sources.py" in text
    assert "scripts.qa_dataset" in text
    assert '"-m", "scripts.qa_dataset"' in text
    assert "training.train" in text
    assert "training.calibrate_thresholds" in text
    assert "training.evaluate" in text
    assert "package_hf_model.py" in text
    assert "torch.cuda.is_available()" in text
    assert "sentencepiece" in text
    assert "Tokenizer preflight: PASS" in text
    assert "TRAINING STDERR" in text
    assert "drive.mount(\"/content/drive\"" in text
    assert "TRAINING_SIGNATURE" in text
    assert "training_complete.json" in text
    assert "resume_model.pt" in text
    assert "Source cache: HIT" in text
    assert "Dataset cache: HIT" in text
    assert "Checkpoint cache: HIT" in text
    assert "Evaluation cache: HIT" in text
    assert "HfApi" in text
    assert "package_ready_hf_write_token_required" in text
    assert "does not have permission to create/write this model repository" in text


def test_shared_project_files_do_not_embed_personal_names():
    blocked_terms = ["sahar", "mahmoud"]
    paths = [
        Path("README.md"),
        Path("docs/PLATFORM_INTEGRATION_GUIDE.md"),
        Path("docs/PRESENTATION_REVISION_GUIDE.md"),
        Path("docs/QUALITY_ASSURANCE_CHECKLIST.md"),
        Path("docs/RELEASE_VALIDATION_STATUS.md"),
        Path("notebooks/model_training.ipynb"),
    ]
    combined = "\n".join(p.read_text(encoding="utf-8").lower() for p in paths)
    for term in blocked_terms:
        assert term not in combined
