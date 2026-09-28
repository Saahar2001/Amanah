import json
from pathlib import Path


def test_showcase_notebook_is_no_training_publish_flow():
    path = Path("notebooks/publish_showcase.ipynb")
    nb = json.loads(path.read_text(encoding="utf-8"))
    text = "\n".join("".join(cell.get("source", [])) for cell in nb["cells"])

    assert 'SHOWCASE_RELEASE = "2026.09.28-s1"' in text
    assert "scripts.build_hf_showcase" in text
    assert "semantic_integrity_model_cache" in text
    assert "evaluation_metrics.json" in text
    assert "metrics_overview.png" in text
    assert "per_label_f1.png" in text
    assert "label_distribution.png" in text
    assert "REMOTE SHOWCASE VERIFICATION: PASS" in text
    assert "training.train" not in text
    assert "CUDA" not in text


def test_model_card_contains_measured_results():
    text = Path("MODEL_CARD.md").read_text(encoding="utf-8")
    assert "94.41%" in text
    assert "93.75%" in text
    assert "92.15%" in text
    assert "5.11%" in text
    assert "3,323" in text
    assert "32,621" in text
