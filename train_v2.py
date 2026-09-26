"""Fresh v2 encoder-decoder pretraining from a streaming raw corpus.

Only pretraining is exposed here. SFT is a separate later phase.
"""
import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from config.seq2seq_model_config import Seq2SeqModelConfig
from data.corpus_pipeline import iter_pretraining_shards
from data.tokenizer import Tokenizer
from model.encoder_decoder import EncoderDecoderLLM


def parse_args():
    p = argparse.ArgumentParser(description="Pretrain LLM v2 from data/raw.")
    p.add_argument("--stage", choices=("pretrain",), default="pretrain")
    p.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    p.add_argument("--tokenizer", type=Path, default=Path("data/processed/tokenizer.json"))
    p.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/v2"))
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--max-steps", type=int, default=10_000)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true", help="Resume model, optimizer, step and completed shards.")
    return p.parse_args()


def device_from_arg(value):
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device("cuda" if value == "auto" and torch.cuda.is_available() else "cpu" if value == "auto" else value)


def _encode_document(text, tokenizer, context_length):
    ids = tokenizer.encode(text).ids
    if len(ids) < 2:
        return
    for start in range(0, len(ids) - 1, context_length):
        chunk = ids[start:start + context_length + 1]
        if len(chunk) < 2:
            break
        split = max(1, min(context_length // 2, len(chunk) - 1))
        encoder_ids = chunk[:split]
        target = chunk[split:]
        if target:
            bos = tokenizer.token_to_id["<bos>"]
            yield encoder_ids, [bos] + target[:-1], target


def _pad_batch(examples, pad_id):
    max_enc = max(len(x[0]) for x in examples)
    max_dec = max(len(x[1]) for x in examples)
    enc, dec, labels = [], [], []
    for e, d, y in examples:
        enc.append(e + [pad_id] * (max_enc - len(e)))
        dec.append(d + [pad_id] * (max_dec - len(d)))
        labels.append(y + [-100] * (max_dec - len(y)))
    return (torch.tensor(enc, dtype=torch.long), torch.tensor(dec, dtype=torch.long), torch.tensor(labels, dtype=torch.long))


def example_stream(documents, tokenizer, context_length):
    for text in documents:
        yield from _encode_document(text, tokenizer, context_length)


def next_batch(stream, pad_id):
    try:
        return _pad_batch([next(stream)], pad_id)
    except StopIteration:
        return None


def loss_for_batch(model, batch, device):
    enc, dec, labels = batch
    logits = model(enc.to(device), dec.to(device))
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.to(device).reshape(-1), ignore_index=-100)


def save_checkpoint(path, model, optimizer, step, shard_index, completed_shards):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "step": step, "shard_index": shard_index, "completed_shards": completed_shards, "validation_loss": math.inf}, path)


def save_progress(path, step, shard_index, completed_shards):
    path.write_text(json.dumps({"step": step, "shard_index": shard_index, "completed_shards": completed_shards}, indent=2), encoding="utf-8")


def load_resume(path, model, optimizer, device):
    state = torch.load(path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    optimizer.load_state_dict(state["optimizer_state_dict"])
    return int(state["step"]), int(state.get("shard_index", 0)), list(state.get("completed_shards", []))


def main():
    args = parse_args()
    if args.batch_size != 1:
        raise ValueError("Streaming mode currently uses batch_size=1; increase effective batch with gradient accumulation.")
    torch.manual_seed(args.seed)
    device = device_from_arg(args.device)
    config = Seq2SeqModelConfig()
    tokenizer = Tokenizer.from_file(args.tokenizer)
    if len(tokenizer) != config.vocab_size:
        raise ValueError(f"Tokenizer vocabulary ({len(tokenizer)}) does not match model vocabulary ({config.vocab_size})")
    pad_id = tokenizer.token_to_id["<pad>"]
    model = EncoderDecoderLLM(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    step, start_shard, completed = 0, 0, []
    latest = args.checkpoint_dir / "latest_model.pt"
    progress = args.checkpoint_dir / "progress.json"
    if args.resume and latest.exists():
        step, start_shard, completed = load_resume(latest, model, optimizer, device)
        print(f"Resuming from step={step:,}, next shard index={start_shard}")

    shards = list(iter_pretraining_shards(args.raw_root))
    if not shards:
        raise FileNotFoundError(f"No pretraining shards found under {args.raw_root}")
    if start_shard >= len(shards):
        print("All raw shards are already complete.")
        return

    print("Stage: pretrain")
    print(f"Raw corpus: {args.raw_root}")
    print(f"Shards: {len(shards)}")
    print(f"Device: {device}")
    print(f"Parameters: {model.num_parameters():,}")
    print("Storage mode: stream one shard at a time; no processed corpus copy")
    print(f"Effective batch size: {args.gradient_accumulation_steps}")

    optimizer.zero_grad(set_to_none=True)
    for shard_index in range(start_shard, len(shards)):
        shard_name, documents = shards[shard_index]
        print(f"\n=== shard {shard_index + 1}/{len(shards)}: {shard_name} ===")
        stream = example_stream(documents, tokenizer, config.context_length)
        while step < args.max_steps:
            accumulated = 0.0
            consumed = 0
            for _ in range(args.gradient_accumulation_steps):
                batch = next_batch(stream, pad_id)
                if batch is None:
                    break
                loss = loss_for_batch(model, batch, device)
                (loss / args.gradient_accumulation_steps).backward()
                accumulated += loss.item()
                consumed += 1
            if consumed == 0:
                break
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if step == 1 or step % 25 == 0:
                print(f"step={step:,} train_loss={accumulated / consumed:.4f}")
            if step % args.eval_every == 0:
                save_checkpoint(latest, model, optimizer, step, shard_index, completed)
                save_progress(progress, step, shard_index, completed)
                print("checkpoint saved")
        if step >= args.max_steps:
            save_checkpoint(latest, model, optimizer, step, shard_index, completed)
            save_progress(progress, step, shard_index, completed)
            print("max_steps reached; stopping.")
            return
        completed.append(shard_name)
        next_index = shard_index + 1
        save_checkpoint(latest, model, optimizer, step, next_index, completed)
        save_progress(progress, step, next_index, completed)
        print(f"Shard complete and checkpointed: {shard_name}")

    save_checkpoint(args.checkpoint_dir / "pretrained_model.pt", model, optimizer, step, len(shards), completed)
    save_progress(progress, step, len(shards), completed)
    print("All pretraining shards complete. SFT is not run by this command.")


if __name__ == "__main__":
    main()
