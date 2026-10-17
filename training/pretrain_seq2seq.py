"""Train a fresh encoder-decoder model on a Parquet corpus using span-denoising."""
import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data.denoising_dataset import DenoisingParquetDataset
from data.tokenizer import Tokenizer
from model.seq2seq.config import Seq2SeqConfig
from model.seq2seq.model import Seq2SeqLLM
from data.parquet_dataset import row_to_text, _split_for_row


def count_usable_tokens(parquet_path, tokenizer, context_length):
    """Count rows/tokens exactly as the denoising dataset sees this Parquet."""
    import pyarrow.parquet as pq

    counts = {"train": 0, "validation": 0, "test": 0}
    rows = {"train": 0, "validation": 0, "test": 0}
    parquet = pq.ParquetFile(parquet_path)
    row_index = 0
    for batch in parquet.iter_batches(batch_size=1024):
        for row in batch.to_pylist():
            split = _split_for_row(row_index, row)
            row_index += 1
            text = row_to_text(row)
            if not text:
                continue
            token_ids = tokenizer.encode(text, add_eos=True)
            if len(token_ids) < 8:
                continue
            counts[split] += min(len(token_ids), context_length)
            rows[split] += 1
    return counts, rows


def collate(batch, pad_id):
    sources, decoder_inputs, targets = zip(*batch)
    source_len = max(map(len, sources))
    target_len = max(map(len, targets))
    source = torch.full((len(batch), source_len), pad_id, dtype=torch.long)
    decoder = torch.full((len(batch), target_len), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), target_len), -100, dtype=torch.long)
    source_mask = torch.zeros((len(batch), source_len), dtype=torch.bool)
    for i, (src, dec, tgt) in enumerate(zip(sources, decoder_inputs, targets)):
        source[i, :len(src)] = src
        decoder[i, :len(dec)] = dec
        labels[i, :len(tgt)] = tgt
        source_mask[i, :len(src)] = True
    return source, decoder, labels, source_mask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, default=Path("data/raw/stage 1/train.parquet"))
    parser.add_argument("--tokenizer", type=Path, default=Path("data/processed/tokenizer.json"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/seq2seq_pretrain"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int, default=10, help="Safety cap on corpus passes")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--embedding-dim", type=int, default=None, help="Override auto-scaled model width")
    parser.add_argument("--auto-architecture", action="store_true", help="Search width/depth/head candidates under the token-derived parameter budget")
    parser.add_argument("--encoder-layers", type=int, default=2)
    parser.add_argument("--decoder-layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--max-train-batches", type=int, default=None, help="Debug/smoke-test limit per pass")
    parser.add_argument("--checkpoint-every-steps", type=int, default=100)
    parser.add_argument("--target-tokens-per-parameter", type=float, default=20.0,
                        help="Approximate token-budget heuristic; not a universal law")
    parser.add_argument("--target-data-passes", type=float, default=2.0,
                        help="Planned repetitions of the unique training split")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if min(args.epochs, args.batch_size, args.gradient_accumulation_steps, args.checkpoint_every_steps) <= 0:
        raise ValueError("epochs, batch size, accumulation, and checkpoint interval must be positive")
    if args.target_tokens_per_parameter <= 0:
        raise ValueError("target tokens per parameter must be positive")
    if args.target_data_passes <= 0:
        raise ValueError("target data passes must be positive")
    if args.heads <= 0 or args.encoder_layers <= 0 or args.decoder_layers <= 0:
        raise ValueError("heads and encoder/decoder layers must be positive")
    if args.heads <= 0 or args.encoder_layers <= 0 or args.decoder_layers <= 0:
        raise ValueError("heads and encoder/decoder layer counts must be positive")
    if args.max_train_batches is not None and args.max_train_batches <= 0:
        raise ValueError("max-train-batches must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device("cuda" if args.device == "cuda" or
                          (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    if not args.parquet.exists():
        raise FileNotFoundError(f"Parquet not found: {args.parquet}")

    tokenizer = Tokenizer.from_file(args.tokenizer)
    usable_tokens, usable_rows = count_usable_tokens(args.parquet, tokenizer, args.context_length)
    train_tokens = usable_tokens["train"]
    if train_tokens <= 0:
        raise ValueError("No usable training tokens found; inspect Parquet schema/tokenizer")
    planned_tokens = math.ceil(train_tokens * args.target_data_passes)
    parameter_budget = planned_tokens / args.target_tokens_per_parameter
    def estimate_parameters(width, encoder_layers, decoder_layers, vocab, context):
        # Approximation for this implementation; the instantiated model count is printed below.
        encoder = encoder_layers * (12 * width * width + 13 * width)
        decoder = decoder_layers * (16 * width * width + 19 * width)
        return (vocab + context) * width + encoder + decoder + 4 * width

    vocab_size = len(tokenizer)
    if args.auto_architecture and args.embedding_dim is not None:
        raise ValueError("Do not combine --auto-architecture with --embedding-dim")
    if args.auto_architecture:
        candidates = []
        for heads in (4, 8):
            for width in range(32, 385, 16):
                if width % heads:
                    continue
                for enc_layers in (2, 4, 6):
                    for dec_layers in (2, 4, 6):
                        params = estimate_parameters(width, enc_layers, dec_layers, vocab_size, args.context_length)
                        if params <= parameter_budget:
                            candidates.append((params, width, heads, enc_layers, dec_layers))
        if candidates:
            _, embedding_dim, selected_heads, selected_enc_layers, selected_dec_layers = max(candidates)
        else:
            # Vocabulary embeddings can exceed the heuristic budget even at the smallest model.
            options = [(estimate_parameters(32, 2, 2, vocab_size, args.context_length), 32, 4, 2, 2)]
            _, embedding_dim, selected_heads, selected_enc_layers, selected_dec_layers = options[0]
    else:
        selected_heads, selected_enc_layers, selected_dec_layers = args.heads, args.encoder_layers, args.decoder_layers
        if args.embedding_dim is None:
            widths = [width for width in range(32, 385, 16) if width % selected_heads == 0]
            fitting = [width for width in widths if estimate_parameters(width, selected_enc_layers, selected_dec_layers, vocab_size, args.context_length) <= parameter_budget]
            embedding_dim = max(fitting) if fitting else min(widths)
        else:
            embedding_dim = args.embedding_dim
            if embedding_dim % selected_heads:
                raise ValueError("embedding-dim must be divisible by heads")

    config = Seq2SeqConfig(
        vocab_size=vocab_size, context_length=args.context_length,
        embedding_dim=embedding_dim, num_heads=selected_heads,
        num_encoder_layers=selected_enc_layers, num_decoder_layers=selected_dec_layers,
        dropout=0.1,
    )
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    model = Seq2SeqLLM(config).to(device)
    parameter_count = sum(p.numel() for p in model.parameters())
    token_budget = planned_tokens
    print(f"Device={device} | Parameters={parameter_count:,} | Vocab={config.vocab_size:,}")
    print(f"Usable train tokens/pass={train_tokens:,} | planned tokens={planned_tokens:,} ({args.target_data_passes:g} passes) | rows={usable_rows['train']:,}")
    print(f"Reference parameter budget={parameter_budget:,.0f} at {args.target_tokens_per_parameter:g} tokens/parameter (heuristic)")
    print(f"Context={config.context_length} | Width={config.embedding_dim} | Encoder={config.num_encoder_layers} | Decoder={config.num_decoder_layers} | Heads={config.num_heads} | Auto-architecture={args.auto_architecture}")
    if parameter_count > parameter_budget:
        print("WARNING: embedding/model minimum exceeds the heuristic parameter budget.")
    print(f"Target training exposure={token_budget:,} tokens ({args.target_data_passes:g} corpus passes); actual tokens/parameter={token_budget / parameter_count:.2f}")
    print("This is span-denoising pretraining, not decoder-only next-token pretraining.")

    train_ds = DenoisingParquetDataset(args.parquet, tokenizer, config.context_length, "train", seed=args.seed)
    val_ds = DenoisingParquetDataset(args.parquet, tokenizer, config.context_length, "validation", seed=args.seed)
    pad_id = tokenizer.token_to_id["<pad>"]
    def make_loader(ds):
        return DataLoader(ds, batch_size=args.batch_size, collate_fn=lambda batch: collate(batch, pad_id))

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    latest = args.checkpoint_dir / "latest.pt"
    best_path = args.checkpoint_dir / "best_model.pt"
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.learning_rate * 0.1
    )
    start_epoch = 0
    resume_batch = 0
    global_step = 0
    tokens_seen = 0
    best_validation_loss = math.inf
    if args.resume:
        if not latest.exists():
            raise FileNotFoundError(f"Cannot resume; checkpoint not found: {latest}")
        checkpoint = torch.load(latest, map_location=device, weights_only=False)
        if checkpoint.get("config") != config.__dict__:
            raise ValueError("Checkpoint config mismatch; use the same architecture/context settings")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint["epoch"])
        resume_batch = int(checkpoint.get("batch_index", 0))
        global_step = int(checkpoint["step"])
        tokens_seen = int(checkpoint.get("tokens_seen", 0))
        best_validation_loss = float(checkpoint.get("best_validation_loss", math.inf))
        print(f"Resuming epoch={start_epoch}, after batch={resume_batch}, tokens_seen={tokens_seen:,}")

    def save_checkpoint(path, epoch, batch_index, validation_loss=None):
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": config.__dict__,
            "epoch": epoch,
            "batch_index": batch_index,
            "step": global_step,
            "tokens_seen": tokens_seen,
            "target_token_budget": token_budget,
            "target_data_passes": args.target_data_passes,
            "best_validation_loss": best_validation_loss,
            "validation_loss": validation_loss,
        }, path)

    def evaluate():
        model.eval()
        loss_sum = 0.0
        token_sum = 0
        with torch.no_grad():
            for batch_index, (src, dec, labels, source_mask) in enumerate(make_loader(val_ds)):
                logits = model(src.to(device), dec.to(device), source_mask.to(device))
                labels = labels.to(device)
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, config.vocab_size), labels.reshape(-1), ignore_index=-100
                )
                valid_tokens = int((labels != -100).sum())
                loss_sum += loss.item() * valid_tokens
                token_sum += valid_tokens
                if batch_index >= 199:
                    break
        if token_sum == 0:
            raise ValueError("Validation split produced no usable tokens; check the dataset and split")
        return loss_sum / token_sum

    history = args.checkpoint_dir / "history.jsonl"
    start_loop_epoch = start_epoch if args.resume and resume_batch > 0 else start_epoch + 1
    with history.open("a", encoding="utf-8") as log:
        for epoch in range(start_loop_epoch, args.epochs + 1):
            train_ds.seed = args.seed + epoch
            model.train()
            losses = []
            epoch_tokens = 0
            epoch_started = time.time()
            optimizer.zero_grad(set_to_none=True)
            pending = 0
            skipped_until = resume_batch if args.resume and epoch == start_epoch else 0

            for batch_index, (src, dec, labels, source_mask) in enumerate(make_loader(train_ds), start=1):
                if batch_index <= skipped_until:
                    continue
                if args.max_train_batches is not None and batch_index - skipped_until > args.max_train_batches:
                    break
                src, dec = src.to(device), dec.to(device)
                labels, source_mask = labels.to(device), source_mask.to(device)
                logits = model(src, dec, source_mask)
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, config.vocab_size), labels.reshape(-1), ignore_index=-100
                )
                (loss / args.gradient_accumulation_steps).backward()
                losses.append(loss.item())
                valid_tokens = int((labels != -100).sum())
                epoch_tokens += valid_tokens
                tokens_seen += valid_tokens
                pending += 1
                if pending == args.gradient_accumulation_steps:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    pending = 0
                    if global_step % args.checkpoint_every_steps == 0:
                        save_checkpoint(latest, epoch, batch_index)
                if batch_index % 10 == 0:
                    print(f"epoch={epoch} batch={batch_index} step={global_step} loss={loss.item():.4f} tokens={tokens_seen:,}/{token_budget:,}")
                if tokens_seen >= token_budget:
                    break

            if pending:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
            if not losses:
                if tokens_seen >= token_budget:
                    break
                raise ValueError("Training produced no batches; inspect data split and resume position")

            validation_loss = evaluate()
            scheduler.step()
            mean_train_loss = sum(losses) / len(losses)
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                save_checkpoint(best_path, epoch, 0, validation_loss)
            save_checkpoint(latest, epoch, 0, validation_loss)
            log.write(json.dumps({
                "epoch": epoch, "step": global_step, "tokens_seen": tokens_seen,
                "epoch_target_tokens": epoch_tokens, "target_token_budget": token_budget,
                "parameters": parameter_count, "train_loss": mean_train_loss,
                "validation_loss": validation_loss, "best_validation_loss": best_validation_loss,
                "seconds": time.time() - epoch_started,
            }) + "\n")
            log.flush()
            print(f"epoch={epoch} train_loss={mean_train_loss:.4f} validation_loss={validation_loss:.4f} tokens={tokens_seen:,}/{token_budget:,}")
            args.resume = False
            resume_batch = 0
            if tokens_seen >= token_budget:
                print("Target-token budget reached.")
                break

if __name__ == "__main__":
    main()
