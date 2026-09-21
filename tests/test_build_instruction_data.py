from data.build_instruction_data import (
    build_records,
    clean_output_dir,
    deduplicate_records,
    split,
    stable_id,
    validate_unique_records,
    verify_written_dataset,
    write_jsonl,
)


def test_stable_id_is_deterministic():
    record = {"instruction": "x", "input": "y", "response": "z", "category": "c", "source": "s"}
    assert stable_id(record) == stable_id(record.copy())


def test_builder_emits_source_grounded_records(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    text = (
        "What is diabetes?\n"
        "Diabetes is a chronic metabolic condition involving elevated blood glucose levels.\n\n"
        "A sufficiently long medical passage explains the cardiovascular system and blood vessels in a neutral way. "
        "The passage contains enough text to exercise the deterministic paragraph extraction path without relying on a generated answer.\n"
    )
    (source / "doc.txt").write_text(text, encoding="utf-8")
    records = build_records(source)
    assert records
    assert all(r["source"] == "doc.txt" for r in records)
    assert all(r["response"] for r in records)
    assert all("id" in r for r in records)
    validate_unique_records(records)


def test_builder_removes_duplicate_examples_after_normalization(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    passage = (
        "This long medical passage explains blood pressure and cardiovascular health "
        "in neutral language and contains enough information for deterministic extraction."
    )
    text = f"{passage}\n\n{passage}  \n"
    (source / "doc.txt").write_text(text, encoding="utf-8")

    records = build_records(source)
    keys = [(r["instruction"], r["input"], r["response"]) for r in records]
    assert len(keys) == len(set(keys))
    assert len(records) == 3


def test_deduplicate_records_is_safe_across_categories_and_sources():
    base = {
        "instruction": "Explain this passage.",
        "input": "A sufficiently long source passage.",
        "response": "A sufficiently long source passage.",
        "category": "grounded_explanation",
        "source": "a.txt",
    }
    duplicate = dict(base, category="grounded_response", source="b.txt")
    duplicate["id"] = stable_id(duplicate)
    base["id"] = stable_id(base)
    unique = deduplicate_records([base, duplicate])
    assert len(unique) == 1
    assert unique[0]["source"] == "a.txt"


def test_split_is_deterministic_and_disjoint(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    text = "A long medical passage about the human heart and circulation. " * 20
    (source / "a.txt").write_text(text, encoding="utf-8")
    records = build_records(source)
    train, validation, test = split(records)
    ids = [set(r["id"] for r in part) for part in (train, validation, test)]
    assert ids[0].isdisjoint(ids[1])
    assert ids[0].isdisjoint(ids[2])
    assert ids[1].isdisjoint(ids[2])
    assert len(train) + len(validation) + len(test) == len(records)


def test_output_cleanup_and_round_trip_validation(tmp_path):
    output = tmp_path / "instruction"
    output.mkdir()
    (output / "train.jsonl").write_text("stale\n", encoding="utf-8")
    (output / "validation.jsonl").write_text("stale\n", encoding="utf-8")
    (output / "test.jsonl").write_text("stale\n", encoding="utf-8")
    (output / "manifest.json").write_text("stale", encoding="utf-8")

    clean_output_dir(output)
    assert not list(output.glob("*.jsonl"))
    assert not (output / "manifest.json").exists()

    records = [
        {
            "instruction": "Explain.",
            "input": "Source passage.",
            "response": "Source passage.",
            "category": "grounded_explanation",
            "source": "doc.txt",
            "id": "one",
        },
        {
            "instruction": "Extract.",
            "input": "Another source passage.",
            "response": "Another source passage.",
            "category": "grounded_extraction",
            "source": "doc.txt",
            "id": "two",
        },
    ]
    write_jsonl(records, output / "train.jsonl")
    write_jsonl([], output / "validation.jsonl")
    write_jsonl([], output / "test.jsonl")
    # The verifier validates the generated split files and their global uniqueness.
    assert verify_written_dataset(output) == 2
