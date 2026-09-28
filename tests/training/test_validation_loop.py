import torch
from torch.utils.data import DataLoader


def test_mean_validation_loss_uses_all_validation_batches():
    from training.train import mean_validation_loss
    class BatchModel(torch.nn.Module):
        def forward(self, input_ids, attention_mask=None, drift_labels=None, severity_labels=None):
            loss = input_ids[:, 0].float().mean()
            return type('Out', (), {'loss': loss})()
    rows=[
        {'input_ids':torch.tensor([1]),'attention_mask':torch.tensor([1]),'drift_labels':torch.tensor([0.]),'severity_labels':torch.tensor(0)},
        {'input_ids':torch.tensor([3]),'attention_mask':torch.tensor([1]),'drift_labels':torch.tensor([0.]),'severity_labels':torch.tensor(0)},
    ]
    loss=mean_validation_loss(BatchModel(),DataLoader(rows,batch_size=1,shuffle=False),torch.device('cpu'),mixed_precision=False)
    assert loss == 2.0


def test_optimizer_flushes_partial_gradient_accumulation_at_epoch_end():
    from training.train import should_optimizer_step
    assert should_optimizer_step(step=2,total_steps=3,gradient_accumulation=2) is True
    assert should_optimizer_step(step=3,total_steps=3,gradient_accumulation=2) is True
    assert should_optimizer_step(step=1,total_steps=3,gradient_accumulation=2) is False


def test_resume_checkpoint_records_epoch_and_uses_half_precision(tmp_path):
    from training.train import _save_resume_checkpoint
    model=torch.nn.Linear(4,2)
    _save_resume_checkpoint(
        model,
        tmp_path,
        completed_epoch=2,
        best_validation_loss=0.5,
        epochs_without_improvement=1,
    )
    state=torch.load(tmp_path/'resume_model.pt',map_location='cpu')
    assert state['weight'].dtype == torch.float16
    meta=__import__('json').loads((tmp_path/'resume_state.json').read_text(encoding='utf-8'))
    assert meta['completed_epoch']==2
    assert meta['best_validation_loss']==0.5
    assert meta['epochs_without_improvement']==1


def test_resume_paths_accept_checkpoint_directory(tmp_path):
    from training.train import _resume_paths
    model_path,state_path=_resume_paths(str(tmp_path))
    assert model_path == tmp_path/'resume_model.pt'
    assert state_path == tmp_path/'resume_state.json'
