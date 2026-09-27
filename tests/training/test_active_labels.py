import json


def test_v0_trainable_labels_only_include_generated_classes():
    from data.models import TRAINABLE_DRIFT_LABELS_V0
    names=[x.value for x in TRAINABLE_DRIFT_LABELS_V0]
    assert names==['FAITHFUL','NEGATION_FLIP','OMISSION','MODALITY_SHIFT','QUANTIFIER_CHANGE','CONDITION_LOSS']
    assert 'TERM_FLATTENING' not in names


def test_training_dataset_uses_v0_active_label_space(tmp_path):
    import torch
    from training.train import JsonlDataset
    class Tok:
        def __call__(self,*args,**kwargs): return {'input_ids':torch.tensor([[1,2]]),'attention_mask':torch.tensor([[1,1]])}
    p=tmp_path/'rows.jsonl'; p.write_text(json.dumps({'source_ar':'x','candidate_en':'y','labels':['FAITHFUL'],'severity':'S0'})+'\n')
    ds=JsonlDataset(p,Tok())
    assert [x.value for x in ds.labels]==['FAITHFUL','NEGATION_FLIP','OMISSION','MODALITY_SHIFT','QUANTIFIER_CHANGE','CONDITION_LOSS']


def test_compute_drift_pos_weight_upweights_rare_positive_labels():
    from training.train import compute_drift_pos_weight
    rows=[{'labels':['FAITHFUL']},{'labels':['FAITHFUL']},{'labels':['FAITHFUL']},{'labels':['NEGATION_FLIP']}]
    weights=compute_drift_pos_weight(rows,['FAITHFUL','NEGATION_FLIP','OMISSION'])
    assert weights[0]==1.0; assert weights[1]==3.0; assert weights[2]==1.0
