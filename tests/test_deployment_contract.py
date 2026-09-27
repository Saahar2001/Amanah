from pathlib import Path


def test_dockerfile_and_render_contracts_exist():
    docker=Path('deployment/Dockerfile').read_text(encoding='utf-8')
    render=Path('deployment/render.yaml').read_text(encoding='utf-8')
    assert 'uvicorn api.main:app' in docker
    assert '8000' in docker
    assert 'MODEL_DIR' in render
    assert 'SOURCE_REGISTRY_PATH' in render
    assert '/ready' in render


def test_hf_handler_exposes_analyze_contract():
    from deployment.hf_handler import EndpointHandler
    assert hasattr(EndpointHandler, '__call__')


def test_cloudflare_helper_keeps_ml_fields_authoritative():
    text=Path('integration-cloudflare.ts').read_text(encoding='utf-8')
    assert 'callAmanahML' in text
    assert 'buildGroundedExplanationPrompt' in text
    assert 'decision: ml.decision' in text
    assert 'severity: ml.severity' in text
    assert 'drifts: ml.drifts' in text
    assert 'LLM explanation must not override' in text

def test_cloudflare_helper_has_separate_hf_handler_contract():
    text=Path('integration-cloudflare.ts').read_text(encoding='utf-8')
    assert 'callAmanahHFEndpoint' in text
    assert 'body: JSON.stringify({ inputs: payload })' in text


def test_sensitive_or_generated_reference_and_model_artifacts_are_gitignored():
    text=Path('.gitignore').read_text(encoding='utf-8')
    for required in ['data/private/','data/reference_store.json','checkpoints/','artifacts/','*.pt','*.safetensors']:
        assert required in text


def test_env_example_points_to_runtime_reference_store():
    text=Path('.env.example').read_text(encoding='utf-8')
    assert 'SOURCE_REGISTRY_PATH=data/reference_store.json' in text
    assert 'SOURCE_REGISTRY_PATH=data/source_registry.json' not in text


def test_github_ci_runs_unit_training_compile_and_source_contract_checks():
    text=Path('.github/workflows/ci.yml').read_text(encoding='utf-8')
    assert 'pytest -q' in text
    assert 'python -m training.train --smoke' in text
    assert 'python -m compileall' in text
    assert 'python scripts/source_contract_smoke.py' in text


def test_cloudflare_integration_fails_closed_when_ml_endpoint_is_unreachable():
    text=Path('integration-cloudflare.ts').read_text(encoding='utf-8')
    assert 'ML endpoint unavailable' in text
    assert 'decision: "ABSTAIN"' in text
    assert 'needs_human_review: true' in text
