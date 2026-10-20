"""Stream the PubMed configuration of armanc/scientific_papers into a resumable text corpus.

This script handles only the second source. It does not modify or rebuild any
existing medical corpus.
"""

import json
import os
from pathlib import Path

from datasets import load_dataset

DATASET_ID = "armanc/scientific_papers"
CONFIG = "pubmed"
SPLIT = "train"
REVISION = "refs/convert/parquet"

OUTPUT_DIR = Path("data/raw")
OUTPUT_FILE = OUTPUT_DIR / "scientific_papers_pubmed.txt"
STATE_FILE = Path("data/processed/scientific_papers_pubmed.progress.json")
CHECKPOINT_EVERY = 100


def _save_state(
    source_rows_scanned: int,
    rows_written: int,
    byte_offset: int,
    complete: bool = False,
) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATE_FILE.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(
            {
                "dataset": DATASET_ID,
                "config": CONFIG,
                "split": SPLIT,
                "revision": REVISION,
                "source_rows_scanned": source_rows_scanned,
                "rows_written": rows_written,
                "byte_offset": byte_offset,
                "complete": complete,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, STATE_FILE)


def _load_state() -> dict:
    if not STATE_FILE.exists():
        return {}
    try:
        state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}

    expected = (DATASET_ID, CONFIG, SPLIT, REVISION)
    actual = (
        state.get("dataset"),
        state.get("config"),
        state.get("split"),
        state.get("revision"),
    )
    return state if actual == expected else {}


def prepare_scientific_papers_pubmed() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    state = _load_state()

    if state.get("complete") and OUTPUT_FILE.exists():
        print(
            f"{DATASET_ID}/{CONFIG}: already complete; "
            f"rows={state.get('rows_written', 0):,}; output={OUTPUT_FILE}"
        )
        return

    source_rows_scanned = int(state.get("source_rows_scanned", 0))
    rows_written = int(state.get("rows_written", 0))
    byte_offset = int(state.get("byte_offset", 0))

    if OUTPUT_FILE.exists():
        # Discard bytes written after the last durable progress checkpoint.
        with OUTPUT_FILE.open("r+b") as output:
            output.truncate(byte_offset)
    else:
        source_rows_scanned = 0
        rows_written = 0
        byte_offset = 0

    print(
        f"Loading {DATASET_ID} config={CONFIG!r} split={SPLIT!r} "
        f"revision={REVISION!r} in streaming mode"
    )
    stream = load_dataset(
        DATASET_ID,
        CONFIG,
        split=SPLIT,
        streaming=True,
        revision=REVISION,
    )

    # Streaming resume skips source rows, including rows with empty text.
    iterator = iter(stream)
    for _ in range(source_rows_scanned):
        try:
            next(iterator)
        except StopIteration as exc:
            raise RuntimeError(
                "The dataset ended before the saved resume position. "
                "Check the saved progress state before restarting from zero."
            ) from exc

    mode = "ab" if OUTPUT_FILE.exists() else "wb"
    pending = 0
    with OUTPUT_FILE.open(mode) as output:
        for row in iterator:
            source_rows_scanned += 1
            article = (row.get("article") or "").strip()
            abstract = (row.get("abstract") or "").strip()

            if article or abstract:
                if article and abstract:
                    block = f"{article}\n\n{abstract}\n\n"
                else:
                    block = f"{article or abstract}\n\n"
                output.write(block.encode("utf-8", errors="replace"))
                rows_written += 1
                pending += 1

            if pending >= CHECKPOINT_EVERY:
                output.flush()
                os.fsync(output.fileno())
                byte_offset = output.tell()
                _save_state(
                    source_rows_scanned, rows_written, byte_offset
                )
                pending = 0
                print(
                    f"Source rows scanned: {source_rows_scanned:,}; "
                    f"records written: {rows_written:,}"
                )

        output.flush()
        os.fsync(output.fileno())
        byte_offset = output.tell()

    _save_state(
        source_rows_scanned, rows_written, byte_offset, complete=True
    )
    print(
        "Scientific papers PubMed corpus complete: "
        f"source_rows={source_rows_scanned:,}; records={rows_written:,}; "
        f"bytes={byte_offset:,}; output={OUTPUT_FILE}"
    )


if __name__ == "__main__":
    prepare_scientific_papers_pubmed()
