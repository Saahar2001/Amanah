import json
from pathlib import Path


def test_training_notebook_runs_clean_pipeline():
    path = Path("notebooks/training_validation.ipynb")
    nb = json.loads(path.read_text(encoding="utf-8"))
    text = "\n".join("".join(cell.get("source", [])) for cell in nb["cells"])

    assert "REPOSITORY_ID = 1391259908" in text
    assert "YOUR_REPO_URL" not in text
    assert "pip\", \"install\", \"-q\", \"-r\", \"requirements.txt" not in text
    assert "fetch_verified_sources.py" in text
    assert "qa_dataset.py" in text
    assert "training.train" in text
    assert "training.calibrate_thresholds" in text
    assert "training.evaluate" in text
    assert "package_hf_model.py" in text
    assert "torch.cuda.is_available()" in text
    assert "HfApi" in text


def test_shared_project_files_do_not_embed_personal_names():
    blocked_terms = ["sahar", "mahmoud"]
    paths = [
        Path("README.md"),
        Path("docs/PLATFORM_INTEGRATION_GUIDE.md"),
        Path("docs/PRESENTATION_REVISION_GUIDE.md"),
        Path("docs/QUALITY_ASSURANCE_CHECKLIST.md"),
        Path("docs/RELEASE_VALIDATION_STATUS.md"),
        Path("notebooks/training_validation.ipynb"),
    ]
    combined = "\n".join(p.read_text(encoding="utf-8").lower() for p in paths)
    for term in blocked_terms:
        assert term not in combined
