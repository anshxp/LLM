"""Training entry point for base, healthcare, and instruction SFT datasets."""
import argparse
import math
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from config.model_config import ModelConfig
from data.dataset import LanguageModelDataset
from data.instruction_dataset import DEFAULT_SFT_CATEGORIES, InstructionDataset, collate_instruction_batch, load_jsonl
from data.instruction_v2_dataset import ShardedInstructionDataset
from data.prepare_training_data import load_token_ids
from evaluation.evaluate import evaluate
from model.llm import LLM
from training.checkpoint import load_checkpoint, save_checkpoint
from training.loss import language_model_loss
from training.optimizer import create_optimizer

DEFAULT_BATCH_SIZE=1
DEFAULT_GRADIENT_ACCUMULATION_STEPS=4
DEFAULT_LEARNING_RATE=3e-4
DEFAULT_WEIGHT_DECAY=0.01
DEFAULT_EPOCHS=10
DEFAULT_LOG_EVERY=10
DEFAULT_MAX_TRAIN_BATCHES=None
DEFAULT_MAX_GRAD_NORM=1.0
DEFAULT_CHECKPOINT_DIR=Path("checkpoints")
DEFAULT_TRAIN_STRIDE=128
DEFAULT_EVAL_STRIDE=256
DEFAULT_LR_MIN=3e-5
DEFAULT_EARLY_STOPPING_PATIENCE=2
DEFAULT_INSTRUCTION_DIR=Path("data/instruction")
DEFAULT_SFT_LEARNING_RATE=5e-5
DEFAULT_SFT_LR_MIN=5e-6
DEFAULT_PRETRAIN_CHECKPOINT=Path("checkpoints/phase7_run/best_model.pt")
DEFAULT_INSTRUCTION_CATEGORIES=",".join(sorted(DEFAULT_SFT_CATEGORIES))

def parse_args(args=None):
    parser=argparse.ArgumentParser(description="Train the healthcare-focused language model.")
    parser.add_argument("--dataset",choices=("base","healthcare","instruction"),default="base")
    parser.add_argument("--batch-size",type=int,default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--gradient-accumulation-steps",type=int,default=DEFAULT_GRADIENT_ACCUMULATION_STEPS)
    parser.add_argument("--learning-rate",type=float,default=None)
    parser.add_argument("--weight-decay",type=float,default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--epochs",type=int,default=DEFAULT_EPOCHS)
    parser.add_argument("--log-every",type=int,default=DEFAULT_LOG_EVERY)
    parser.add_argument("--max-train-batches",type=int,default=DEFAULT_MAX_TRAIN_BATCHES)
    parser.add_argument("--max-grad-norm",type=float,default=DEFAULT_MAX_GRAD_NORM)
    parser.add_argument("--checkpoint-dir",type=Path,default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--resume",type=Path,default=None)
    parser.add_argument("--device",choices=("auto","cpu","cuda"),default="auto")
    parser.add_argument("--num-workers",type=int,default=0)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--train-stride",type=int,default=DEFAULT_TRAIN_STRIDE)
    parser.add_argument("--eval-stride",type=int,default=DEFAULT_EVAL_STRIDE)
    parser.add_argument("--lr-min",type=float,default=None)
    parser.add_argument("--early-stopping-patience",type=int,default=DEFAULT_EARLY_STOPPING_PATIENCE)
    parser.add_argument("--pretrained-checkpoint",type=Path,default=None)
    parser.add_argument("--instruction-dir",type=Path,default=DEFAULT_INSTRUCTION_DIR)
    parser.add_argument("--instruction-categories",default=DEFAULT_INSTRUCTION_CATEGORIES)
    parsed=parser.parse_args(args)
    if parsed.learning_rate is None: parsed.learning_rate=DEFAULT_SFT_LEARNING_RATE if parsed.dataset=="instruction" else DEFAULT_LEARNING_RATE
    if parsed.lr_min is None: parsed.lr_min=DEFAULT_SFT_LR_MIN if parsed.dataset=="instruction" else DEFAULT_LR_MIN
    if parsed.dataset=="instruction" and parsed.pretrained_checkpoint is None: parsed.pretrained_checkpoint=DEFAULT_PRETRAIN_CHECKPOINT
    parsed.instruction_categories=tuple(x.strip() for x in parsed.instruction_categories.split(",") if x.strip())
    return parsed

def resolve_device(requested):
    if requested=="cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA was requested but is not available")
    if requested=="cpu": return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def validate_args(args):
    if args.batch_size<=0 or args.gradient_accumulation_steps<=0 or args.epochs<=0: raise ValueError("batch-size, gradient-accumulation-steps and epochs must be positive")
    if args.learning_rate<=0 or args.lr_min<=0 or args.lr_min>args.learning_rate: raise ValueError("invalid learning-rate/lr-min")
    if args.weight_decay<0 or args.num_workers<0: raise ValueError("invalid weight decay or worker count")

def build_dataset(split,dataset_name,context_length,stride,instruction_dir=None,instruction_categories=None):
    if dataset_name=="instruction":
        instruction_dir=Path(instruction_dir or DEFAULT_INSTRUCTION_DIR)
        if list(instruction_dir.glob(f"{split}-*.jsonl")):
            return ShardedInstructionDataset(instruction_dir,split,context_length,categories=instruction_categories or None)
        records=load_jsonl(instruction_dir/f"{split}.jsonl",categories=instruction_categories or DEFAULT_SFT_CATEGORIES)
        return InstructionDataset(records,context_length=context_length)
    token_ids=load_token_ids(split,dataset=dataset_name)
    return LanguageModelDataset(token_ids=token_ids,context_length=context_length,stride=stride)

def make_loader(dataset,dataset_name,batch_size,shuffle,num_workers):
    kwargs={"batch_size":batch_size,"shuffle":shuffle,"num_workers":num_workers}
    if dataset_name=="instruction": kwargs["collate_fn"]=collate_instruction_batch
    return DataLoader(dataset,**kwargs)

def main(args=None):
    args=parse_args(args); validate_args(args); torch.manual_seed(args.seed)
    device=resolve_device(args.device); config=ModelConfig()
    train_dataset=build_dataset("train",args.dataset,config.context_length,args.train_stride,args.instruction_dir,args.instruction_categories)
    validation_dataset=build_dataset("validation",args.dataset,config.context_length,args.eval_stride,args.instruction_dir,args.instruction_categories)
    if not len(train_dataset) or not len(validation_dataset): raise ValueError("Training and validation splits must contain complete sequences")
    train_loader=make_loader(train_dataset,args.dataset,args.batch_size,True,args.num_workers)
    validation_loader=make_loader(validation_dataset,args.dataset,args.batch_size,False,args.num_workers)
    print(f"Dataset: {args.dataset} | train={len(train_dataset):,} | validation={len(validation_dataset):,}")
    model=LLM(config).to(device); print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    optimizer=create_optimizer(model,args.learning_rate,args.weight_decay)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=max(1,args.epochs),eta_min=args.lr_min)
    args.checkpoint_dir.mkdir(parents=True,exist_ok=True)
    start_epoch=global_step=0; best_validation_loss=math.inf; epochs_without_improvement=0
    if args.pretrained_checkpoint is not None and args.resume is not None: raise ValueError("Use either --pretrained-checkpoint or --resume, not both")
    if args.pretrained_checkpoint is not None:
        checkpoint=torch.load(args.pretrained_checkpoint,map_location=device,weights_only=False)
        state_dict=checkpoint.get("model_state_dict",checkpoint.get("model"))
        if state_dict is None: raise ValueError("Pretrained checkpoint does not contain model_state_dict or model")
        model.load_state_dict(state_dict); print(f"Loaded pretrained model weights from {args.pretrained_checkpoint}")
    if args.resume is not None:
        state=load_checkpoint(model,optimizer,args.resume,map_location=device,scheduler=scheduler)
        global_step=state["step"]; start_epoch=state["epoch"]
        if state.get("best_validation_loss") is not None: best_validation_loss=float(state["best_validation_loss"])
        epochs_without_improvement=int(state.get("epochs_without_improvement",0))
    best_path=args.checkpoint_dir/"best_model.pt"
    for epoch in range(start_epoch+1,args.epochs+1):
        model.train(); optimizer.zero_grad(set_to_none=True); accumulation=0; running=0.0
        for batch_index,(input_ids,target_ids) in enumerate(train_loader):
            if args.max_train_batches is not None and batch_index>=args.max_train_batches: break
            loss=language_model_loss(model(input_ids.to(device)),target_ids.to(device)); (loss/args.gradient_accumulation_steps).backward()
            running+=loss.item(); accumulation+=1
            update=accumulation==args.gradient_accumulation_steps or batch_index+1==len(train_loader) or (args.max_train_batches is not None and batch_index+1>=args.max_train_batches)
            if update:
                if accumulation<args.gradient_accumulation_steps:
                    scale=args.gradient_accumulation_steps/accumulation
                    for p in model.parameters():
                        if p.grad is not None: p.grad.mul_(scale)
                torch.nn.utils.clip_grad_norm_(model.parameters(),args.max_grad_norm); optimizer.step(); optimizer.zero_grad(set_to_none=True); global_step+=1; accumulation=0
                if global_step==1 or global_step%args.log_every==0: print(f"Epoch {epoch}/{args.epochs} | Step {global_step} | Loss {running/max(1,args.gradient_accumulation_steps):.4f}"); running=0.0
        metrics=evaluate(model,validation_loader,device=device); validation_loss=metrics["loss"]; print(f"Validation loss: {validation_loss:.4f} | Perplexity: {metrics['perplexity']:.2f}")
        improved=validation_loss<best_validation_loss
        if improved: best_validation_loss=validation_loss; epochs_without_improvement=0
        else: epochs_without_improvement+=1
        scheduler.step(); checkpoint_path=args.checkpoint_dir/f"model_epoch_{epoch}.pt"
        save_checkpoint(model,optimizer,global_step,checkpoint_path,epoch=epoch,scheduler=scheduler,best_validation_loss=best_validation_loss,epochs_without_improvement=epochs_without_improvement)
        if improved: save_checkpoint(model,optimizer,global_step,best_path,epoch=epoch,scheduler=scheduler,best_validation_loss=best_validation_loss,epochs_without_improvement=epochs_without_improvement)
        if args.early_stopping_patience>0 and epochs_without_improvement>=args.early_stopping_patience: break
    print(f"Training complete. Best validation loss: {best_validation_loss:.4f}")
    if args.dataset=="instruction":
        test_dataset=build_dataset("test",args.dataset,config.context_length,args.eval_stride,args.instruction_dir,args.instruction_categories)
        test_metrics=evaluate(model,make_loader(test_dataset,args.dataset,args.batch_size,False,args.num_workers),device=device)
        print(f"Instruction test loss: {test_metrics['loss']:.4f} | Perplexity: {test_metrics['perplexity']:.2f}")

if __name__=="__main__": main()
