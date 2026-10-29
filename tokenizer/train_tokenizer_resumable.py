"""Experimental checkpointable BPE trainer backed by SQLite.

Unlike tokenizers.BpeTrainer, this implementation persists its own BPE state:
pre-tokenized piece frequencies, pair counts, ordered merges, and vocabulary.
It is designed to prove resume correctness on small corpora before large runs.

Warning: exact full-corpus BPE can be computationally expensive. Benchmark on a
representative subset before attempting a multi-million-document corpus.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from datasets import load_dataset
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers


SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[BOS]", "[EOS]", "[SYSTEM]", "[USER]", "[ASSISTANT]"]


def meta_get(db: sqlite3.Connection, key: str, default: str = "") -> str:
    row = db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return row[0] if row else default


def meta_set(db: sqlite3.Connection, key: str, value: object) -> None:
    db.execute(
        "INSERT INTO meta(key,value) VALUES(?,?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, str(value)),
    )


def pair_counts(symbols: list[str]) -> Counter[tuple[str, str]]:
    return Counter(zip(symbols, symbols[1:]))


def merge_pair(symbols: list[str], left: str, right: str) -> list[str]:
    merged: list[str] = []
    i = 0
    while i < len(symbols):
        if i + 1 < len(symbols) and symbols[i] == left and symbols[i + 1] == right:
            merged.append(left + right)
            i += 2
        else:
            merged.append(symbols[i])
            i += 1
    return merged


def open_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=120)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=FULL")
    db.execute("PRAGMA foreign_keys=ON")
    db.executescript("""
        CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS pieces(
            id INTEGER PRIMARY KEY,
            piece TEXT NOT NULL UNIQUE,
            freq INTEGER NOT NULL,
            symbols TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS pair_counts(
            left_symbol TEXT NOT NULL,
            right_symbol TEXT NOT NULL,
            count INTEGER NOT NULL,
            PRIMARY KEY(left_symbol,right_symbol)
        );
        CREATE TABLE IF NOT EXISTS piece_pairs(
            piece_id INTEGER NOT NULL,
            left_symbol TEXT NOT NULL,
            right_symbol TEXT NOT NULL,
            occurrences INTEGER NOT NULL,
            PRIMARY KEY(piece_id,left_symbol,right_symbol),
            FOREIGN KEY(piece_id) REFERENCES pieces(id)
        );
        CREATE INDEX IF NOT EXISTS ix_piece_pairs_pair
            ON piece_pairs(left_symbol,right_symbol,piece_id);
        CREATE TABLE IF NOT EXISTS vocab(
            token TEXT PRIMARY KEY,
            token_id INTEGER NOT NULL UNIQUE
        );
        CREATE TABLE IF NOT EXISTS merges(
            rank INTEGER PRIMARY KEY,
            left_symbol TEXT NOT NULL,
            right_symbol TEXT NOT NULL,
            merged_token TEXT NOT NULL
        );
    """)
    db.commit()
    return db


def add_pair_delta(db: sqlite3.Connection, left: str, right: str, delta: int) -> None:
    row = db.execute(
        "SELECT count FROM pair_counts WHERE left_symbol=? AND right_symbol=?",
        (left, right),
    ).fetchone()
    new_count = (row[0] if row else 0) + delta
    if new_count <= 0:
        db.execute(
            "DELETE FROM pair_counts WHERE left_symbol=? AND right_symbol=?",
            (left, right),
        )
    elif row:
        db.execute(
            "UPDATE pair_counts SET count=? WHERE left_symbol=? AND right_symbol=?",
            (new_count, left, right),
        )
    else:
        db.execute(
            "INSERT INTO pair_counts(left_symbol,right_symbol,count) VALUES(?,?,?)",
            (left, right, new_count),
        )


def initialise_vocab(db: sqlite3.Connection, alphabet: list[str]) -> None:
    if db.execute("SELECT COUNT(*) FROM vocab").fetchone()[0]:
        return
    with db:
        token_id = 0
        for token in SPECIAL_TOKENS + alphabet:
            if db.execute("SELECT 1 FROM vocab WHERE token=?", (token,)).fetchone():
                continue
            db.execute("INSERT INTO vocab(token,token_id) VALUES(?,?)", (token, token_id))
            token_id += 1
        meta_set(db, "phase", "ingest")
        meta_set(db, "rows_seen", 0)
        meta_set(db, "documents", 0)
        meta_set(db, "characters", 0)
        meta_set(db, "pair_cursor", 0)
        meta_set(db, "merge_count", 0)


def ingest_corpus(db: sqlite3.Connection, args, normalizer, pre_tokenizer) -> None:
    rows_seen = int(meta_get(db, "rows_seen", "0"))
    documents = int(meta_get(db, "documents", "0"))
    characters = int(meta_get(db, "characters", "0"))
    print(f"[resume] Corpus ingestion resumes after {rows_seen:,} raw rows.", flush=True)

    dataset = load_dataset(args.dataset, split=args.split, streaming=True)
    if rows_seen:
        dataset = dataset.skip(rows_seen)

    batch_piece_freq: Counter[str] = Counter()
    batch_rows = batch_docs = batch_chars = 0
    started = time.monotonic()
    last_log = started

    def flush() -> None:
        nonlocal rows_seen, documents, characters, batch_rows, batch_docs, batch_chars
        if batch_rows == 0:
            return
        with db:
            for piece, amount in batch_piece_freq.items():
                symbols = json.dumps(list(piece), ensure_ascii=False)
                db.execute(
                    "INSERT INTO pieces(piece,freq,symbols) VALUES(?,?,?) "
                    "ON CONFLICT(piece) DO UPDATE SET freq=freq+excluded.freq",
                    (piece, amount, symbols),
                )
            rows_seen += batch_rows
            documents += batch_docs
            characters += batch_chars
            meta_set(db, "rows_seen", rows_seen)
            meta_set(db, "documents", documents)
            meta_set(db, "characters", characters)
        batch_piece_freq.clear()
        batch_rows = batch_docs = batch_chars = 0

    for row in dataset:
        batch_rows += 1
        value = row.get(args.text_column)
        if isinstance(value, str) and value.strip():
            normalized = normalizer.normalize_str(value)
            for piece, _offsets in pre_tokenizer.pre_tokenize_str(normalized):
                if piece:
                    batch_piece_freq[piece] += 1
            batch_docs += 1
            batch_chars += len(value)
        if batch_rows >= args.ingest_batch_rows:
            flush()
        if args.max_rows is not None and rows_seen + batch_rows >= args.max_rows:
            flush()
            print(f"[ingest] Reached --max-rows={args.max_rows:,}; finishing bounded test corpus.", flush=True)
            break
        now = time.monotonic()
        if now - last_log >= args.log_every_seconds:
            print(
                f"[ingest] raw_rows={rows_seen + batch_rows:,} "
                f"documents={documents + batch_docs:,} "
                f"unique_batch_pieces={len(batch_piece_freq):,} "
                f"elapsed_min={(now-started)/60:.1f}",
                flush=True,
            )
            last_log = now
    flush()
    with db:
        meta_set(db, "phase", "count_pairs")
    print(
        f"[ingest-done] documents={meta_get(db, 'documents')} "
        f"characters={meta_get(db, 'characters')} "
        f"unique_pieces={db.execute('SELECT COUNT(*) FROM pieces').fetchone()[0]:,}",
        flush=True,
    )


def initialise_pair_counts(db: sqlite3.Connection, batch_size: int = 2000) -> None:
    cursor = int(meta_get(db, "pair_cursor", "0"))
    total = db.execute("SELECT COUNT(*) FROM pieces").fetchone()[0]
    print(f"[resume] Pair-count initialisation resumes after piece id {cursor:,}.", flush=True)
    while True:
        rows = db.execute(
            "SELECT id,freq,symbols FROM pieces WHERE id>? ORDER BY id LIMIT ?",
            (cursor, batch_size),
        ).fetchall()
        if not rows:
            break
        with db:
            for piece_id, freq, symbols_json in rows:
                symbols = json.loads(symbols_json)
                counts = pair_counts(symbols)
                for (left, right), occurrences in counts.items():
                    db.execute(
                        "INSERT INTO piece_pairs(piece_id,left_symbol,right_symbol,occurrences) "
                        "VALUES(?,?,?,?)",
                        (piece_id, left, right, occurrences),
                    )
                    db.execute(
                        "INSERT INTO pair_counts(left_symbol,right_symbol,count) VALUES(?,?,?) "
                        "ON CONFLICT(left_symbol,right_symbol) DO UPDATE "
                        "SET count=count+excluded.count",
                        (left, right, occurrences * freq),
                    )
            cursor = rows[-1][0]
            meta_set(db, "pair_cursor", cursor)
        if cursor % 20000 < batch_size:
            print(f"[pair-counts] pieces={cursor:,}/{total:,}", flush=True)
    with db:
        meta_set(db, "phase", "merging")
    print(f"[pair-counts-done] pair_types={db.execute('SELECT COUNT(*) FROM pair_counts').fetchone()[0]:,}", flush=True)


def perform_one_merge(db: sqlite3.Connection, min_frequency: int) -> bool:
    top = db.execute(
        "SELECT left_symbol,right_symbol,count FROM pair_counts "
        "ORDER BY count DESC,left_symbol ASC,right_symbol ASC LIMIT 1"
    ).fetchone()
    if not top or top[2] < min_frequency:
        return False
    left, right, count = top
    merged_token = left + right

    # A single transaction makes each merge atomic. If the VM dies mid-merge,
    # SQLite rolls it back and the same merge can be attempted after restart.
    with db:
        db.execute("DROP TABLE IF EXISTS temp.affected")
        db.execute("CREATE TEMP TABLE affected(piece_id INTEGER PRIMARY KEY)")
        db.execute(
            "INSERT INTO affected SELECT piece_id FROM piece_pairs "
            "WHERE left_symbol=? AND right_symbol=?",
            (left, right),
        )
        affected_count = db.execute("SELECT COUNT(*) FROM affected").fetchone()[0]
        cursor = db.execute("SELECT piece_id FROM affected ORDER BY piece_id")
        for (piece_id,) in cursor:
            row = db.execute(
                "SELECT freq,symbols FROM pieces WHERE id=?", (piece_id,)
            ).fetchone()
            if not row:
                continue
            freq, symbols_json = row
            old_symbols = json.loads(symbols_json)
            old_pairs = pair_counts(old_symbols)
            for (pair_left, pair_right), occurrences in old_pairs.items():
                add_pair_delta(db, pair_left, pair_right, -(occurrences * freq))
            db.execute("DELETE FROM piece_pairs WHERE piece_id=?", (piece_id,))

            new_symbols = merge_pair(old_symbols, left, right)
            db.execute(
                "UPDATE pieces SET symbols=? WHERE id=?",
                (json.dumps(new_symbols, ensure_ascii=False), piece_id),
            )
            for (pair_left, pair_right), occurrences in pair_counts(new_symbols).items():
                db.execute(
                    "INSERT INTO piece_pairs(piece_id,left_symbol,right_symbol,occurrences) "
                    "VALUES(?,?,?,?)",
                    (piece_id, pair_left, pair_right, occurrences),
                )
                add_pair_delta(db, pair_left, pair_right, occurrences * freq)

        next_rank = db.execute("SELECT COALESCE(MAX(rank),0)+1 FROM merges").fetchone()[0]
        db.execute(
            "INSERT INTO merges(rank,left_symbol,right_symbol,merged_token) VALUES(?,?,?,?)",
            (next_rank, left, right, merged_token),
        )
        if not db.execute("SELECT 1 FROM vocab WHERE token=?", (merged_token,)).fetchone():
            next_id = db.execute("SELECT COALESCE(MAX(token_id),-1)+1 FROM vocab").fetchone()[0]
            db.execute("INSERT INTO vocab(token,token_id) VALUES(?,?)", (merged_token, next_id))
        meta_set(db, "merge_count", next_rank)
    print(
        f"[merge] rank={next_rank:,} pair=({left!r},{right!r}) "
        f"count={count:,} affected_pieces={affected_count:,} vocab={db.execute('SELECT COUNT(*) FROM vocab').fetchone()[0]:,}",
        flush=True,
    )
    return True


def export_tokenizer(db: sqlite3.Connection, args, output_dir: Path) -> None:
    vocab = {
        token: token_id
        for token, token_id in db.execute("SELECT token,token_id FROM vocab ORDER BY token_id")
    }
    merges = [
        (left, right)
        for left, right in db.execute(
            "SELECT left_symbol,right_symbol FROM merges ORDER BY rank"
        )
    ]
    tokenizer = Tokenizer(models.BPE(vocab=vocab, merges=merges, unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer_path = output_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))

    probes = [
        "pneumothorax", "pleural effusion", "atelectasis", "cardiomegaly",
        "pulmonary edema", "consolidation", "ground-glass opacity",
        "No focal air-space consolidation or pleural effusion.",
    ]
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "split": args.split,
        "algorithm": "custom-checkpointable-BPE",
        "target_vocab_size": args.vocab_size,
        "actual_vocab_size": len(vocab),
        "documents_consumed": int(meta_get(db, "documents", "0")),
        "characters_consumed": int(meta_get(db, "characters", "0")),
        "raw_rows_consumed": int(meta_get(db, "rows_seen", "0")),
        "merge_count": len(merges),
        "min_frequency": args.min_frequency,
        "checkpoint_database": str(args.state_db),
        "probe_tokenization": {
            text: {"tokens": tokenizer.encode(text).tokens, "ids": tokenizer.encode(text).ids}
            for text in probes
        },
        "note": (
            "Custom deterministic BPE implementation. Validate against project-specific "
            "tokenization expectations before using it for model training."
        ),
    }
    (output_dir / "tokenizer_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[done] Saved tokenizer: {tokenizer_path}", flush=True)
    print(f"[done] Saved report: {output_dir / 'tokenizer_report.json'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="common-pile/pubmed")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--state-db", type=Path, required=True,
                        help="Persistent SQLite checkpoint database; keep it on durable storage.")
    parser.add_argument("--vocab-size", type=int, default=300000)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Optional raw-row limit for smoke tests; omit for full corpus")
    parser.add_argument("--ingest-batch-rows", type=int, default=50)
    parser.add_argument("--log-every-seconds", type=int, default=30)
    args = parser.parse_args()

    if args.vocab_size < 256 or args.min_frequency < 1:
        parser.error("vocab size must be >=256 and min frequency must be positive")
    if args.ingest_batch_rows < 1 or args.log_every_seconds < 1:
        parser.error("batch and log intervals must be positive")
    if args.max_rows is not None and args.max_rows < 1:
        parser.error("--max-rows must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    db = open_db(args.state_db)
    normalizer = normalizers.NFKC()
    pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    alphabet = sorted(pre_tokenizer.alphabet())
    initialise_vocab(db, alphabet)

    # Refuse to resume a state database against a different input definition.
    # A larger --vocab-size is allowed so an existing run can continue further.
    run_config = {
        "dataset": args.dataset,
        "split": args.split,
        "text_column": args.text_column,
        "normalizer": "NFKC",
        "pre_tokenizer": "ByteLevel(add_prefix_space=False)",
    }
    for key, value in run_config.items():
        stored = meta_get(db, "config_" + key)
        if stored and stored != value:
            db.close()
            parser.error(
                f"Checkpoint config mismatch for {key}: stored={stored!r}, requested={value!r}. "
                "Use the original settings or a new --state-db."
            )
        if not stored:
            with db:
                meta_set(db, "config_" + key, value)

    try:
        phase = meta_get(db, "phase", "ingest")
        if phase == "ingest":
            ingest_corpus(db, args, normalizer, pre_tokenizer)
            phase = meta_get(db, "phase", "ingest")
        if phase == "count_pairs":
            initialise_pair_counts(db)
            phase = meta_get(db, "phase", "ingest")
        if phase == "merging":
            print(
                f"[resume] merge_count={meta_get(db, 'merge_count','0')} "
                f"vocab_size={db.execute('SELECT COUNT(*) FROM vocab').fetchone()[0]:,}",
                flush=True,
            )
            while db.execute("SELECT COUNT(*) FROM vocab").fetchone()[0] < args.vocab_size:
                if not perform_one_merge(db, args.min_frequency):
                    print("[stop] No pair meets min_frequency; vocabulary target not reached.", flush=True)
                    break
        export_tokenizer(db, args, args.output_dir)
    finally:
        db.close()


if __name__ == "__main__":
    main()
