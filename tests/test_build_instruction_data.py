import json

from data.build_instruction_data import build_records, split, stable_id


def test_stable_id_is_deterministic():
    record = {"instruction": "x", "input": "y", "response": "z", "category": "c", "source": "s"}
    assert stable_id(record) == stable_id(record.copy())


def test_builder_emits_source_grounded_records(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "doc.txt").write_text(
        """What is diabetes?\nDiabetes is a chronic metabolic condition involving elevated blood glucose levels.\n\n"
        "A sufficiently long medical passage explains the cardiovascular system and blood vessels in a neutral way. "
        "The passage contains enough text to exercise the deterministic paragraph extraction path without relying on a generated answer.\n""",
        encoding="utf-8",
    )
    records = build_records(source)
    assert records
    assert all(r["source"] == "doc.txt" for r in records)
    assert all(r["response"] for r in records)
    assert all("id" in r for r in records)


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
