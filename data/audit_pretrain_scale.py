"""Count actual tokenizer tokens and estimate the Chinchilla-style parameter reference."""
import argparse, math
from pathlib import Path
import pyarrow.parquet as pq
from data.parquet_dataset import row_to_text, _split_for_row
from data.tokenizer import Tokenizer

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parquet",type=Path,default=Path("data/raw/stage 1/train.parquet"))
    p.add_argument("--tokenizer",type=Path,default=Path("data/processed/tokenizer.json"))
    p.add_argument("--target-tokens-per-parameter",type=float,default=20.0)
    a=p.parse_args()
    if not a.parquet.exists(): raise FileNotFoundError(a.parquet)
    tok=Tokenizer.from_file(a.tokenizer); counts={"train":0,"validation":0,"test":0}; rows={"train":0,"validation":0,"test":0}
    parquet=pq.ParquetFile(a.parquet)
    for batch in parquet.iter_batches(batch_size=1024):
        for row in batch.to_pylist():
            split=_split_for_row(sum(rows.values()),row)
            text=row_to_text(row)
            if text:
                counts[split]+=len(tok.encode(text,add_eos=True)); rows[split]+=1
    train_tokens=counts["train"]
    print(f"file_mb={a.parquet.stat().st_size/(1024*1024):.2f} vocab={len(tok):,}")
    for split in ("train","validation","test"):
        print(f"{split}: rows={rows[split]:,} tokens={counts[split]:,}")
    print(f"reference_parameter_budget={train_tokens/a.target_tokens_per_parameter:,.0f} at {a.target_tokens_per_parameter:g} tokens/parameter (heuristic)")
    print("Architecture embedding floor may limit matching this target; use held-out loss to validate choices.")
if __name__=="__main__": main()
