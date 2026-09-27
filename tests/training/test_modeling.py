import torch
from torch import nn


class TinyOutput:
    def __init__(self, hidden):
        self.last_hidden_state = hidden


class TinyEncoder(nn.Module):
    def __init__(self, hidden_size=8, vocab=32):
        super().__init__()
        self.config = type('Cfg', (), {'hidden_size': hidden_size})()
        self.emb = nn.Embedding(vocab, hidden_size)
    def forward(self, input_ids, attention_mask=None, **kwargs):
        return TinyOutput(self.emb(input_ids))


def test_multitask_model_shapes_and_losses():
    from training.modeling import AmanahMultiTaskConfig, AmanahMultiTaskModel
    cfg=AmanahMultiTaskConfig(num_drift_labels=5,num_severity_labels=4,dropout=0.0)
    model=AmanahMultiTaskModel(TinyEncoder(),cfg)
    ids=torch.tensor([[1,2,3],[4,5,0]]); mask=torch.tensor([[1,1,1],[1,1,0]])
    drift_labels=torch.tensor([[1,0,0,0,0],[0,1,0,1,0]],dtype=torch.float32); severity_labels=torch.tensor([0,3])
    out=model(input_ids=ids,attention_mask=mask,drift_labels=drift_labels,severity_labels=severity_labels)
    assert out.drift_logits.shape==(2,5); assert out.severity_logits.shape==(2,4); assert out.loss is not None and torch.isfinite(out.loss)


def test_masked_mean_pooling_ignores_padding():
    from training.modeling import AmanahMultiTaskConfig, AmanahMultiTaskModel
    model=AmanahMultiTaskModel(TinyEncoder(hidden_size=4),AmanahMultiTaskConfig(num_drift_labels=2,num_severity_labels=4,dropout=0.0))
    out=model(input_ids=torch.tensor([[1,2,3]]),attention_mask=torch.tensor([[1,1,0]])); assert out.drift_logits.shape==(1,2)


def test_checkpoint_metadata_embeds_encoder_config_for_offline_loading():
    from training.train import checkpoint_metadata
    encoder_cfg=type('Cfg',(),{'to_dict':lambda self:{'model_type':'bert','hidden_size':8}})()
    fake_model=type('M',(),{'encoder':type('E',(),{'config':encoder_cfg})()})()
    meta=checkpoint_metadata(fake_model,model_name='base',seed=42,best_validation_loss=1.25)
    assert meta['base_model']=='base'; assert meta['encoder_config']=={'model_type':'bert','hidden_size':8}; assert meta['seed']==42


def test_multitask_model_accepts_positive_class_weights():
    from training.modeling import AmanahMultiTaskConfig, AmanahMultiTaskModel
    cfg=AmanahMultiTaskConfig(num_drift_labels=2,num_severity_labels=4,dropout=0.0,drift_pos_weight=(1.0,5.0))
    model=AmanahMultiTaskModel(TinyEncoder(),cfg)
    out=model(input_ids=torch.tensor([[1,2,3]]),attention_mask=torch.tensor([[1,1,1]]),drift_labels=torch.tensor([[0.0,1.0]]),severity_labels=torch.tensor([3]))
    assert out.loss is not None and torch.isfinite(out.loss)
