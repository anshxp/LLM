"""Count the token budget actually usable by the seq2seq denoising dataset."""
import argparse
from pathlib import Path
import pyarrow.parquet as pq
from data.parquet_dataset import row_to_text, _split_for_row
from data.tokenizer import Tokenizer

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet",type=Path,default=Path("data/raw/stage 1/train.parquet"))
    parser.add_argument("--tokenizer",type=Path,default=Path("data/processed/tokenizer.json"))
    parser.add_argument("--context-length",type=int,default=512)
    parser.add_argument("--target-tokens-per-parameter",type=float,default=20.0)
    args=parser.parse_args()
    if not args.parquet.exists(): raise FileNotFoundError(args.parquet)
    if args.context_length<8: raise ValueError("context-length must be >= 8")
    if args.target_tokens_per_parameter<=0: raise ValueError("target-tokens-per-parameter must be positive")
    tokenizer=Tokenizer.from_file(args.tokenizer)
    counts={"train":0,"validation":0,"test":0}; rows={"train":0,"validation":0,"test":0}
    parquet=pq.ParquetFile(args.parquet)
    row_index=0
    for batch in parquet.iter_batches(batch_size=1024):
        for row in batch.to_pylist():
            split=_split_for_row(row_index,row); row_index+=1
            text=row_to_text(row)
            if not text: continue
            token_ids=tokenizer.encode(text,add_eos=True)
            if len(token_ids)<8: continue
            # Match the pretraining dataset, which truncates each row to context length.
            counts[split]+=min(len(token_ids),args.context_length); rows[split]+=1
    train_tokens=counts["train"]
    print(f"file_mb={args.parquet.stat().st_size/(1024*1024):.2f} vocab={len(tokenizer):,} context={args.context_length}")
    for split in ("train","validation","test"):
        print(f"{split}: usable_rows={rows[split]:,} usable_target_tokens_per_pass={counts[split]:,}")
    print(f"reference_parameter_budget={train_tokens/args.target_tokens_per_parameter:,.0f} at {args.target_tokens_per_parameter:g} tokens/parameter (heuristic)")
    print("This estimate matches one dataset pass, not repeated-epoch tokens. Embedding size and held-out loss must also constrain the model choice.")
if __name__=="__main__": main()
