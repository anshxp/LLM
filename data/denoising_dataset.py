"""Denoising dataset for encoder-decoder pretraining on Parquet rows."""
from __future__ import annotations
import random, torch
from torch.utils.data import IterableDataset
from data.parquet_dataset import row_to_text,_split_for_row

class DenoisingParquetDataset(IterableDataset):
    def __init__(self,parquet_file,tokenizer,context_length,split="train",noise_density=.15,mean_span_length=3.0,seed=42,max_examples=None):
        if context_length<8: raise ValueError("context_length must be >= 8")
        if not 0<noise_density<.5: raise ValueError("noise_density must be in (0, 0.5)")
        self.parquet_file=str(parquet_file); self.tokenizer=tokenizer; self.context_length=context_length; self.split=split
        self.noise_density=noise_density; self.mean_span_length=mean_span_length; self.seed=seed; self.max_examples=max_examples
        self.total_bytes=__import__("os").path.getsize(self.parquet_file); self.rows_seen=self.tokens_yielded=self.sequences_yielded=self.bytes_read=0
    @property
    def total_mb(self): return self.total_bytes/(1024*1024)
    @property
    def data_consumed_mb(self): return self.bytes_read/(1024*1024)
    @property
    def data_remaining_mb(self): return max(0.,(self.total_bytes-self.bytes_read)/(1024*1024))
    @property
    def progress_fraction(self): return min(1.,self.bytes_read/max(1,self.total_bytes))
    def progress_snapshot(self):
        return {"total_bytes":self.total_bytes,"bytes_read":self.bytes_read,"data_consumed_mb":self.data_consumed_mb,
                "data_remaining_mb":self.data_remaining_mb,"progress_fraction":self.progress_fraction,
                "tokens_yielded":self.tokens_yielded,"sequences_yielded":self.sequences_yielded}
    def __iter__(self):
        import pyarrow.parquet as pq
        rng=random.Random(self.seed); self.rows_seen=self.tokens_yielded=self.sequences_yielded=self.bytes_read=0
        parquet=pq.ParquetFile(self.parquet_file); total_rows=max(1,parquet.metadata.num_rows); seen=0
        for batch in parquet.iter_batches(batch_size=1024):
            for row in batch.to_pylist():
                idx=self.rows_seen; self.rows_seen+=1
                if _split_for_row(idx,row)!=self.split: continue
                text=row_to_text(row)
                if not text: continue
                original=self.tokenizer.encode(text,add_eos=True)
                if len(original)<8: continue
                original=original[:self.context_length]; src=self._corrupt(original,rng); target=original
                decoder=[self.tokenizer.token_to_id["<bos>"]]+target[:-1]
                self.tokens_yielded+=len(target); self.sequences_yielded+=1
                yield torch.tensor(src,dtype=torch.long),torch.tensor(decoder,dtype=torch.long),torch.tensor(target,dtype=torch.long)
                seen+=1
                if self.max_examples is not None and seen>=self.max_examples: return
            self.bytes_read=min(self.total_bytes,int(self.total_bytes*self.rows_seen/total_rows))
    def _corrupt(self,tokens,rng):
        source=list(tokens); target=max(1,int(len(source)*self.noise_density)); masked=set()
        attempts=0
        while len(masked)<target and attempts<len(source)*10:
            attempts+=1; start=rng.randrange(len(source))
            if start in masked: continue
            span=max(1,int(rng.expovariate(1.0/self.mean_span_length)))
            for pos in range(start,min(len(source),start+span)):
                if len(masked)>=target: break
                masked.add(pos)
        unk=self.tokenizer.token_to_id["<unk>"]
        for pos in masked: source[pos]=unk
        return source
