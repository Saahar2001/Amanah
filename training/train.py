from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from data.models import Severity, TRAINABLE_DRIFT_LABELS_V0
from training.modeling import AmanahMultiTaskConfig, AmanahMultiTaskModel

DEFAULT_MODEL = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"


def checkpoint_metadata(model, *, model_name: str, seed: int, best_validation_loss: float, max_length: int = 256, drift_pos_weight: list[float] | None = None) -> dict:
    return {"base_model":model_name,"encoder_config":model.encoder.config.to_dict(),"drift_labels":[x.value for x in TRAINABLE_DRIFT_LABELS_V0],"severity_labels":[x.value for x in Severity],"seed":seed,"best_validation_loss":best_validation_loss,"max_length":max_length,"drift_pos_weight":drift_pos_weight}


def mean_validation_loss(model: nn.Module, loader: DataLoader, device: torch.device, *, mixed_precision: bool) -> float:
    was_training=model.training; model.eval(); losses=[]
    with torch.no_grad():
        for batch in loader:
            batch={k:v.to(device) for k,v in batch.items()}
            with torch.autocast(device_type=device.type,enabled=mixed_precision and device.type=="cuda"): out=model(**batch)
            losses.append(float(out.loss.detach().cpu()))
    if was_training: model.train()
    if not losses: raise ValueError("validation dataset is empty")
    return float(sum(losses)/len(losses))


def compute_drift_pos_weight(rows: list[dict], label_names: list[str], *, max_weight: float = 20.0) -> list[float]:
    total=len(rows)
    if total==0: raise ValueError("training dataset is empty")
    weights=[]
    for name in label_names:
        positives=sum(name in row.get("labels",[]) for row in rows)
        if positives==0: weights.append(1.0); continue
        negatives=total-positives; weights.append(float(min(max_weight,max(1.0,negatives/positives))))
    return weights


def seed_everything(seed:int)->None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def should_optimizer_step(*,step:int,total_steps:int,gradient_accumulation:int)->bool:
    if gradient_accumulation<1: raise ValueError("gradient_accumulation must be >= 1")
    return step%gradient_accumulation==0 or step==total_steps


class JsonlDataset(Dataset):
    def __init__(self,path:str|Path,tokenizer,max_length:int=256):
        self.rows=[json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
        self.tokenizer=tokenizer; self.max_length=max_length; self.labels=list(TRAINABLE_DRIFT_LABELS_V0); self.label_index={x.value:i for i,x in enumerate(self.labels)}; self.severity_index={x.value:i for i,x in enumerate(Severity)}
    def __len__(self): return len(self.rows)
    def __getitem__(self,idx):
        row=self.rows[idx]; tok=self.tokenizer(row["source_ar"],row["candidate_en"],truncation=True,max_length=self.max_length,padding="max_length",return_tensors="pt")
        drift=torch.zeros(len(self.labels),dtype=torch.float32)
        for label in row["labels"]: drift[self.label_index[label]]=1.0
        return {"input_ids":tok["input_ids"].squeeze(0),"attention_mask":tok["attention_mask"].squeeze(0),"drift_labels":drift,"severity_labels":torch.tensor(self.severity_index[row["severity"]])}


def train_checkpoint(*,train_path:str,validation_path:str,output_dir:str,model_name:str=DEFAULT_MODEL,seed:int=42,epochs:int=3,batch_size:int=8,learning_rate:float=2e-5,gradient_accumulation:int=1,mixed_precision:bool=True,resume:str|None=None,max_length:int=256,patience:int=2)->str:
    from transformers import AutoTokenizer
    seed_everything(seed); tokenizer=AutoTokenizer.from_pretrained(model_name); train_ds=JsonlDataset(train_path,tokenizer,max_length=max_length); validation_ds=JsonlDataset(validation_path,tokenizer,max_length=max_length)
    label_names=[x.value for x in TRAINABLE_DRIFT_LABELS_V0]; drift_pos_weight=compute_drift_pos_weight(train_ds.rows,label_names)
    config=AmanahMultiTaskConfig(num_drift_labels=len(TRAINABLE_DRIFT_LABELS_V0),drift_pos_weight=tuple(drift_pos_weight)); model=AmanahMultiTaskModel.from_pretrained_encoder(model_name,config)
    if resume: model.load_state_dict(torch.load(resume,map_location="cpu"))
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); model.to(device); loader=DataLoader(train_ds,batch_size=batch_size,shuffle=True); validation_loader=DataLoader(validation_ds,batch_size=batch_size,shuffle=False); optim=torch.optim.AdamW(model.parameters(),lr=learning_rate); scaler=torch.amp.GradScaler("cuda",enabled=mixed_precision and device.type=="cuda")
    outdir=Path(output_dir); outdir.mkdir(parents=True,exist_ok=True); best_validation_loss=float("inf"); epochs_without_improvement=0; model.train(); optim.zero_grad(set_to_none=True)
    for _ in range(epochs):
        for step,batch in enumerate(loader,start=1):
            batch={k:v.to(device) for k,v in batch.items()}
            with torch.autocast(device_type=device.type,enabled=mixed_precision and device.type=="cuda"): out=model(**batch); loss=out.loss/gradient_accumulation
            scaler.scale(loss).backward()
            if should_optimizer_step(step=step,total_steps=len(loader),gradient_accumulation=gradient_accumulation):
                scaler.unscale_(optim); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); scaler.step(optim); scaler.update(); optim.zero_grad(set_to_none=True)
        validation_loss=mean_validation_loss(model,validation_loader,device,mixed_precision=mixed_precision)
        if validation_loss<best_validation_loss-1e-6:
            best_validation_loss=validation_loss; epochs_without_improvement=0; torch.save(model.state_dict(),outdir/"model.pt")
        else:
            epochs_without_improvement+=1
            if patience>=0 and epochs_without_improvement>=patience: break
    tokenizer.save_pretrained(outdir)
    (outdir/"amanah_config.json").write_text(json.dumps(checkpoint_metadata(model,model_name=model_name,seed=seed,best_validation_loss=best_validation_loss,max_length=max_length,drift_pos_weight=drift_pos_weight),indent=2),encoding="utf-8")
    return str(outdir)


def smoke()->None:
    class TinyOutput:
        def __init__(self,hidden): self.last_hidden_state=hidden
    class TinyEncoder(nn.Module):
        def __init__(self): super().__init__(); self.config=type("Cfg",(),{"hidden_size":8})(); self.emb=nn.Embedding(32,8)
        def forward(self,input_ids,attention_mask=None,**_): return TinyOutput(self.emb(input_ids))
    model=AmanahMultiTaskModel(TinyEncoder(),AmanahMultiTaskConfig(num_drift_labels=len(TRAINABLE_DRIFT_LABELS_V0))); out=model(input_ids=torch.tensor([[1,2,3]]),attention_mask=torch.tensor([[1,1,1]]),drift_labels=torch.zeros((1,len(TRAINABLE_DRIFT_LABELS_V0))),severity_labels=torch.tensor([0])); assert out.loss is not None and torch.isfinite(out.loss); print("AMANAH training smoke: OK")


def main()->None:
    p=argparse.ArgumentParser(); p.add_argument("--train"); p.add_argument("--validation"); p.add_argument("--output-dir",default="checkpoints/amanah-drift-v0.1"); p.add_argument("--model-name",default=DEFAULT_MODEL); p.add_argument("--seed",type=int,default=42); p.add_argument("--epochs",type=int,default=3); p.add_argument("--batch-size",type=int,default=8); p.add_argument("--learning-rate",type=float,default=2e-5); p.add_argument("--gradient-accumulation",type=int,default=1); p.add_argument("--no-mixed-precision",action="store_true"); p.add_argument("--resume"); p.add_argument("--max-length",type=int,default=256); p.add_argument("--patience",type=int,default=2); p.add_argument("--smoke",action="store_true"); args=p.parse_args()
    if args.smoke: smoke(); return
    if not args.train or not args.validation: p.error("--train and --validation are required unless --smoke is used")
    print(train_checkpoint(train_path=args.train,validation_path=args.validation,output_dir=args.output_dir,model_name=args.model_name,seed=args.seed,epochs=args.epochs,batch_size=args.batch_size,learning_rate=args.learning_rate,gradient_accumulation=args.gradient_accumulation,mixed_precision=not args.no_mixed_precision,resume=args.resume,max_length=args.max_length,patience=args.patience))


if __name__=="__main__": main()
