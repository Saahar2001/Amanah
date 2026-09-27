import json
from pathlib import Path


def test_package_hf_repo_contains_custom_handler_checkpoint_and_reference_store(tmp_path):
    from scripts.package_hf_model import package_hf_repo
    checkpoint=tmp_path/'checkpoint'; checkpoint.mkdir()
    (checkpoint/'model.pt').write_bytes(b'model')
    (checkpoint/'amanah_config.json').write_text(json.dumps({'base_model':'x','drift_labels':[],'severity_labels':[]}),encoding='utf-8')
    (checkpoint/'thresholds.json').write_text('[]',encoding='utf-8'); (checkpoint/'tokenizer_config.json').write_text('{}',encoding='utf-8')
    refs=tmp_path/'reference_store.json'; refs.write_text('{"ayahs":[]}',encoding='utf-8')
    out=tmp_path/'hf'; package_hf_repo(checkpoint,refs,out)
    assert (out/'handler.py').exists(); assert (out/'model.pt').read_bytes()==b'model'; assert (out/'reference_store.json').exists(); assert (out/'amanah_engine'/'service.py').exists(); assert (out/'training'/'modeling.py').exists(); assert (out/'data'/'models.py').exists(); assert 'huggingface_hub' in (out/'requirements.txt').read_text(encoding='utf-8')


def test_hf_handler_uses_endpoint_path_for_model_and_reference_store():
    text=Path('deployment/hf_handler.py').read_text(encoding='utf-8')
    assert "ClassifierAdapter(root)" in text
    assert "JsonReferenceStore(root / 'reference_store.json')" in text
    assert 'build_default_service' not in text


def test_packaged_model_card_enables_hf_endpoint_deploy_button(tmp_path):
    from scripts.package_hf_model import package_hf_repo
    checkpoint=tmp_path/'checkpoint'; checkpoint.mkdir(); (checkpoint/'model.pt').write_bytes(b'model'); (checkpoint/'amanah_config.json').write_text(json.dumps({'base_model':'x','drift_labels':[],'severity_labels':[]}),encoding='utf-8')
    refs=tmp_path/'reference_store.json'; refs.write_text('{"ayahs":[]}',encoding='utf-8'); out=tmp_path/'hf'; package_hf_repo(checkpoint,refs,out); card=(out/'README.md').read_text(encoding='utf-8'); assert 'pipeline_tag: text-classification' in card; assert 'endpoints-template' in card
