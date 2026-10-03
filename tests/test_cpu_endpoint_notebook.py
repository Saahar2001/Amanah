import json
from pathlib import Path


def test_cpu_endpoint_notebook_is_low_cost_and_no_training():
    path = Path("notebooks/deploy_cpu_endpoint.ipynb")
    nb = json.loads(path.read_text(encoding="utf-8"))
    text = "\n".join("".join(cell.get("source", [])) for cell in nb["cells"])

    assert "semantic-integrity-v0-1-cpu" in text
    assert 'task="custom"' in text
    assert 'accelerator="cpu"' in text
    assert "min_replica=0" in text
    assert "max_replica=1" in text
    assert "scale_to_zero_timeout=15" in text
    assert 'type="authenticated"' in text
    assert "LIVE ENDPOINT SMOKE TEST: PASS" in text
    assert "AMANAH_ML_URL=" in text
    assert "training.train" not in text
    assert "CUDA" not in text
