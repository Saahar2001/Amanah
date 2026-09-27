from pathlib import Path

def test_github_ci_runs_deterministic_local_checks_only():
    text=Path('.github/workflows/ci.yml').read_text(encoding='utf-8')
    assert 'pytest -q' in text
    assert 'python -m training.train --smoke' in text
    assert 'python -m compileall' in text
    assert 'source_contract_smoke.py' not in text
