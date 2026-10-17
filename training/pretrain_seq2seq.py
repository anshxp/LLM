"""Fresh encoder-decoder span-denoising pretraining."""
import argparse,json,math,random,time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from model.seq2seq.config import Seq2SeqConfig
from data.tokenizer import Tokenizer
from data.denoising_dataset import DenoisingParquetDataset
from model.seq2seq.model import Seq2SeqLLM
def collate(batch,pad_id):
    srcs,decs,tgts=zip(*batch); ms=max(map(len,srcs)); mt=max(map(len,tgts))
    src=torch.full((len(batch),ms),pad_id,dtype=torch.long); dec=torch.full((len(batch),mt),pad_id,dtype=torch.long); tgt=torch.full((len(batch),mt),-100,dtype=torch.long); mask=torch.zeros((len(batch),ms),dtype=torch.bool)
    for i,(s,d,t) in enumerate(batch): src[i,:len(s)]=s; dec[i,:len(d)]=d; tgt[i,:len(t)]=t; mask[i,:len(s)]=True
    return src,dec,tgt,mask
def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--parquet",type=Path,default=Path("data/raw/stage 1/train.parquet")); p.add_argument("--tokenizer",type=Path,default=Path("data/processed/tokenizer.json")); p.add_argument("--checkpoint-dir",type=Path,default=Path("checkpoints/seq2seq_pretrain")); p.add_argument("--device",choices=("auto","cpu","cuda"),default="auto"); p.add_argument("--epochs",type=int,default=10); p.add_argument("--batch-size",type=int,default=1); p.add_argument("--gradient-accumulation-steps",type=int,default=8); p.add_argument("--learning-rate",type=float,default=1e-4); p.add_argument("--weight-decay",type=float,default=.01); p.add_argument("--context-length",type=int,default=512); p.add_argument("--embedding-dim",type=int,default=128); p.add_argument("--encoder-layers",type=int,default=2); p.add_argument("--decoder-layers",type=int,default=2); p.add_argument("--heads",type=int,default=4); p.add_argument("--max-train-batches",type=int,default=None); p.add_argument("--checkpoint-every-steps",type=int,default=100); p.add_argument("--seed",type=int,default=42); p.add_argument("--resume",action="store_true"); a=p.parse_args()
    if a.epochs<1 or a.batch_size<1 or a.gradient_accumulation_steps<1: raise ValueError("training sizes must be positive")
    if a.device=="cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    device=torch.device("cuda" if (a.device=="cuda" or (a.device=="auto" and torch.cuda.is_available())) else "cpu")
    if not a.parquet.exists(): raise FileNotFoundError(f"Parquet not found: {a.parquet}")
    tok=Tokenizer.from_file(a.tokenizer); vocab=len(tok)
    cfg=Seq2SeqConfig(vocab_size=vocab,context_length=a.context_length,embedding_dim=a.embedding_dim,num_heads=a.heads,num_encoder_layers=a.encoder_layers,num_decoder_layers=a.decoder_layers,num_layers=max(a.encoder_layers,a.decoder_layers),dropout=.1)
    torch.manual_seed(a.seed); random.seed(a.seed); model=Seq2SeqLLM(cfg).to(device); params=sum(p.numel() for p in model.parameters())
    print(f"device={device} params={params:,} vocab={vocab:,} context={a.context_length}; 20-token/parameter reference={params*20:,} tokens (heuristic only)")
    train=DenoisingParquetDataset(a.parquet,tok,a.context_length,"train",seed=a.seed); valid=DenoisingParquetDataset(a.parquet,tok,a.context_length,"validation",seed=a.seed); pad=tok.token_to_id["<pad>"]
    def loader(ds): return DataLoader(ds,batch_size=a.batch_size,collate_fn=lambda b:collate(b,pad))
    a.checkpoint_dir.mkdir(parents=True,exist_ok=True); latest=a.checkpoint_dir/"latest.pt"; bestp=a.checkpoint_dir/"best_model.pt"; opt=torch.optim.AdamW(model.parameters(),lr=a.learning_rate,weight_decay=a.weight_decay); sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=a.epochs,eta_min=a.learning_rate*.1); start=0; step=0; best=math.inf
    if a.resume:
        if not latest.exists(): raise FileNotFoundError(f"Cannot resume; missing {latest}")
        ck=torch.load(latest,map_location=device,weights_only=False)
        if ck.get("config")!=cfg.__dict__: raise ValueError("Checkpoint config mismatch; use same settings")
        model.load_state_dict(ck["model_state_dict"]); opt.load_state_dict(ck["optimizer_state_dict"])
        if "scheduler_state_dict" in ck: sched.load_state_dict(ck["scheduler_state_dict"])
        start=int(ck["epoch"]); step=int(ck["step"]); best=float(ck["best_validation_loss"])
    def evaluate():
        model.eval(); total=0.; count=0
        with torch.no_grad():
            for i,(s,d,t,m) in enumerate(loader(valid)):
                logits=model(s.to(device),d.to(device),m.to(device)); labels=t.to(device); loss=torch.nn.functional.cross_entropy(logits.reshape(-1,vocab),labels.reshape(-1),ignore_index=-100); n=int((labels!=-100).sum()); total+=loss.item()*n; count+=n
                if i>=199: break
        if count==0: raise ValueError("No validation examples; check Parquet split")
        return total/count
    hist=a.checkpoint_dir/"history.jsonl"
    with hist.open("a",encoding="utf-8") as f:
        for epoch in range(start+1,a.epochs+1):
            model.train(); losses=[]; tokens=0; started=time.time(); opt.zero_grad(set_to_none=True); pending=0
            for bi,(s,d,t,m) in enumerate(loader(train),1):
                if a.max_train_batches is not None and bi>a.max_train_batches: break
                s,d,labels,m=s.to(device),d.to(device),t.to(device),m.to(device); logits=model(s,d,m); loss=torch.nn.functional.cross_entropy(logits.reshape(-1,vocab),labels.reshape(-1),ignore_index=-100); (loss/a.gradient_accumulation_steps).backward(); losses.append(loss.item()); tokens+=int((labels!=-100).sum()); pending+=1
                if pending==a.gradient_accumulation_steps:
                    torch.nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step(); opt.zero_grad(set_to_none=True); step+=1; pending=0
                    if step%a.checkpoint_every_steps==0: torch.save({"model_state_dict":model.state_dict(),"optimizer_state_dict":opt.state_dict(),"scheduler_state_dict":sched.state_dict(),"config":cfg.__dict__,"epoch":epoch-1,"step":step,"best_validation_loss":best},latest)
                if bi%10==0: print(f"epoch={epoch} batch={bi} step={step} loss={loss.item():.4f} target_tokens={tokens:,}")
            if pending: torch.nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step(); opt.zero_grad(set_to_none=True); step+=1
            val=evaluate(); avg=sum(losses)/max(1,len(losses)); sched.step(); state={"model_state_dict":model.state_dict(),"optimizer_state_dict":opt.state_dict(),"scheduler_state_dict":sched.state_dict(),"config":cfg.__dict__,"epoch":epoch,"step":step,"train_loss":avg,"validation_loss":val,"best_validation_loss":min(best,val),"parameters":params}; torch.save(state,latest)
            if val<best: best=val; state["best_validation_loss"]=best; torch.save(state,bestp)
            f.write(json.dumps({"epoch":epoch,"step":step,"train_loss":avg,"validation_loss":val,"best_validation_loss":best,"target_tokens":tokens,"parameters":params,"seconds":time.time()-started})+"\n"); f.flush(); print(f"epoch={epoch} train_loss={avg:.4f} val_loss={val:.4f} best={best:.4f} tokens={tokens:,}")
if __name__=="__main__": main()
