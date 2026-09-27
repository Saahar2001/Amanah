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
