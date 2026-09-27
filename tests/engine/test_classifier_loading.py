import sys
import types


def test_build_encoder_from_saved_config_does_not_require_remote_model(monkeypatch):
    class FakeConfig:
        @classmethod
        def for_model(cls,model_type,**payload): return types.SimpleNamespace(model_type=model_type,**payload)
    class FakeModel:
        @classmethod
        def from_config(cls,config): return types.SimpleNamespace(config=config)
    monkeypatch.setitem(sys.modules,'transformers',types.SimpleNamespace(AutoConfig=FakeConfig,AutoModel=FakeModel))
    from amanah_engine.classifier import build_encoder_from_saved_config
    encoder=build_encoder_from_saved_config({'model_type':'bert','vocab_size':32,'hidden_size':8,'num_hidden_layers':1,'num_attention_heads':2,'intermediate_size':16})
    assert encoder.config.hidden_size==8 and encoder.config.model_type=='bert'
