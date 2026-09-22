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
    supervised, all_records = build_records(source)
    assert supervised
    assert all_records
    assert all(r["source"] == "doc.txt" for r in all_records)
    assert all(r["response"] for r in supervised)
    assert all("id" in r for r in supervised)
    assert any(r["category"] == "source_qa" for r in supervised)
    assert all(r["category"] != "grounded_response" for r in supervised)
    validate_unique_records(supervised)
    validate_unique_records(all_records)


def test_builder_separates_supervised_and_passage_copy_records(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    passage = (
        "This long medical passage explains blood pressure and cardiovascular health "
        "in neutral language and contains enough information for deterministic extraction."
    )
    text = f"This is a question about diabetes?\nDiabetes is a chronic metabolic condition involving elevated blood glucose levels.\n\n{passage}\n"
    (source / "doc.txt").write_text(text, encoding="utf-8")
    supervised, all_records = build_records(source)
    assert any(r["category"] == "source_qa" for r in supervised)
    assert any(r["category"] == "grounded_explanation" for r in all_records)
    assert len(all_records) > len(supervised)
    assert len(all_records) > len(supervised)


def test_split_is_deterministic_and_disjoint():
    records = []
    for index in range(30):
        records.append(
            {
                "instruction": f"Q{index}",
                "input": f"input-{index}",
                "response": f"response-{index}",
                "category": "source_qa",
                "source": "doc.txt",
                "id": f"{index:04d}",
            }
        )
    first = split(records)
    second = split(records)
    assert first == second
    ids = [set(r["id"] for r in part) for part in first]
    assert ids[0].isdisjoint(ids[1])
    assert ids[0].isdisjoint(ids[2])
    assert ids[1].isdisjoint(ids[2])
    assert len(first[0]) + len(first[1]) + len(first[2]) == len(records)


def test_split_rejects_too_small_dataset():
    records = [
        {"instruction": "Q", "input": "I", "response": "R", "category": "source_qa", "source": "s", "id": "1"},
        {"instruction": "Q2", "input": "I2", "response": "R2", "category": "source_qa", "source": "s", "id": "2"},
    ]
    import pytest

    with pytest.raises(ValueError, match="At least 3"):
        split(records)


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
    assert verify_written_dataset(output) == 2
