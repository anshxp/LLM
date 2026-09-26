"""Train one prepared continued-pretraining shard."""
from __future__ import annotations
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from config.model_config import ModelConfig
from data.continued_pretraining_stream import StreamingTokenDataset, iter_prepared_texts
from data.tokenizer import Tokenizer
from evaluation.evaluate import evaluate
from model.llm import LLM
from training.checkpoint import load_checkpoint, save_checkpoint
from training.loss import language_model_loss
from training.optimizer import create_optimizer

DEFAULT_FOUNDATION=Path("checkpoints/phase7_run/best_model.pt")
DEFAULT_TOKENIZER=Path("data/processed/tokenizer.json")
DEFAULT_CHECKPOINT_DIR=Path("checkpoints/continued_pretraining")

def parse_args(args=None):
 p=argparse.ArgumentParser()
 p.add_argument("--prepared-dir",required=True,type=Path); p.add_argument("--pretrained-checkpoint",type=Path,default=DEFAULT_FOUNDATION)
 p.add_argument("--tokenizer",type=Path,default=DEFAULT_TOKENIZER); p.add_argument("--checkpoint-dir",type=Path,default=DEFAULT_CHECKPOINT_DIR)
 p.add_argument("--shard-index",type=int,required=True); p.add_argument("--shard-name",required=True)
 p.add_argument("--batch-size",type=int,default=1); p.add_argument("--gradient-accumulation-steps",type=int,default=4)
 p.add_argument("--learning-rate",type=float,default=3e-4); p.add_argument("--weight-decay",type=float,default=.01)
 p.add_argument("--checkpoint-every-steps",type=int,default=500); p.add_argument("--log-every",type=int,default=50)
 p.add_argument("--max-grad-norm",type=float,default=1.0); p.add_argument("--device",choices=("auto","cpu","cuda"),default="auto")
 p.add_argument("--num-workers",type=int,default=0); p.add_argument("--reset-run",action="store_true")
 return p.parse_args(args)

def device(name):
 if name=="cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA was requested but is not available")
 return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def loader(path,tokenizer,context,batch,workers):
 ds=StreamingTokenDataset(lambda _split: iter_prepared_texts(path),tokenizer,context_length=context,split="train")
 return DataLoader(ds,batch_size=batch,shuffle=False,num_workers=workers,pin_memory=torch.cuda.is_available())

def load_model(model,opt,checkpoint,foundation,dev,reset):
 if checkpoint.exists() and not reset: return load_checkpoint(model,opt,checkpoint,map_location=dev,restore_rng=True)
 if not foundation.exists(): raise FileNotFoundError(f"Foundation checkpoint not found: {foundation}")
 payload=torch.load(foundation,map_location=dev,weights_only=False); state=payload.get("model_state_dict",payload.get("model"))
 if state is None: raise ValueError("Foundation checkpoint does not contain model weights")
 model.load_state_dict(state); return None

def main(args=None):
 o=parse_args(args); prepared=o.prepared_dir; train_path=prepared/"train.jsonl"; val_path=prepared/"validation.jsonl"
 if not train_path.exists() or not val_path.exists(): raise FileNotFoundError(f"Prepared shard is incomplete: {prepared}")
 dev=device(o.device); cfg=ModelConfig(); tok=Tokenizer.from_file(o.tokenizer)
 if len(tok)!=cfg.vocab_size: raise ValueError("Tokenizer vocabulary does not match model vocabulary")
 o.checkpoint_dir.mkdir(parents=True,exist_ok=True); checkpoint=o.checkpoint_dir/"latest.pt"
 model=LLM(cfg).to(dev); opt=create_optimizer(model,learning_rate=o.learning_rate,weight_decay=o.weight_decay)
 state=load_model(model,opt,checkpoint,o.pretrained_checkpoint,dev,o.reset_run)
 saved_shard=state.get("shard_index",-1) if state else -1; saved_name=state.get("shard_name") if state else None
 completed=state.get("completed_shards",0) if state else 0
 if state and completed>o.shard_index: print(f"Shard {o.shard_index} already complete"); return
 if state and saved_shard==o.shard_index and saved_name not in (None,o.shard_name): raise RuntimeError(f"Checkpoint belongs to shard {saved_name}, not {o.shard_name}")
 global_step=int(state.get("global_step",0)) if state else 0
 resume_batch=int(state.get("batch_index",0)) if state and saved_shard==o.shard_index else 0
 train=loader(train_path,tok,cfg.context_length,o.batch_size,o.num_workers); model.train(); opt.zero_grad(set_to_none=True); accum=0; running=0.0
 for bi,(x,y) in enumerate(train):
  if bi<resume_batch: continue
  x=x.to(dev,non_blocking=True); y=y.to(dev,non_blocking=True); loss=language_model_loss(model(x),y); (loss/o.gradient_accumulation_steps).backward(); running+=loss.item(); accum+=1
  if accum<o.gradient_accumulation_steps: continue
  torch.nn.utils.clip_grad_norm_(model.parameters(),o.max_grad_norm); opt.step(); opt.zero_grad(set_to_none=True); global_step+=1; accum=0
  if global_step==1 or global_step%o.log_every==0: print(f"Shard {o.shard_index} | step={global_step} | batch={bi+1} | loss={running/o.gradient_accumulation_steps:.4f}"); running=0.0
  if global_step%o.checkpoint_every_steps==0:
   save_checkpoint(model,opt,global_step,checkpoint,batch_index=bi+1,extra_state={"shard_index":o.shard_index,"shard_name":o.shard_name,"completed_shards":o.shard_index})
 if accum:
  scale=o.gradient_accumulation_steps/accum
  for p in model.parameters():
   if p.grad is not None: p.grad.mul_(scale)
  torch.nn.utils.clip_grad_norm_(model.parameters(),o.max_grad_norm); opt.step(); opt.zero_grad(set_to_none=True); global_step+=1
 val=loader(val_path,tok,cfg.context_length,o.batch_size,o.num_workers); metrics=evaluate(model,val,device=dev); print(f"Shard validation | loss={metrics['loss']:.4f} perplexity={metrics['perplexity']:.2f}")
 extra={"shard_index":o.shard_index,"shard_name":o.shard_name,"completed_shards":o.shard_index+1,"validation_loss":metrics["loss"],"validation_perplexity":metrics["perplexity"]}
 save_checkpoint(model,opt,global_step,o.checkpoint_dir/f"model_shard_{o.shard_index:05d}.pt",extra_state=extra); save_checkpoint(model,opt,global_step,checkpoint,extra_state=extra); print(f"Completed shard {o.shard_index}: {o.shard_name}")

if __name__=="__main__": main()
