import json

import data.build_corpus as build_module


VALID_TEXT = (
    "Pneumonia is an infection of the lungs. It may be caused by bacteria, "
    "viruses, or other microorganisms. Clinical presentation can vary "
    "depending on the underlying cause. Patients may develop fever, cough, "
    "shortness of breath, chest pain, and fatigue. Diagnosis commonly involves "
    "clinical assessment, laboratory testing, and imaging when appropriate."
)


def test_build_corpus_handles_nested_files_and_content_duplicates(
    tmp_path,
    monkeypatch,
):
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    nested = raw / "nested"
    nested.mkdir(parents=True)

    first = raw / "first.txt"
    duplicate = nested / "duplicate.txt"
    first.write_text(VALID_TEXT, encoding="utf-8")
    duplicate.write_text(VALID_TEXT.replace("  ", " "), encoding="utf-8")

    monkeypatch.setattr(build_module, "RAW_DIR", raw)
    monkeypatch.setattr(build_module, "OUTPUT_FILE", processed / "corpus.txt")
    monkeypatch.setattr(
        build_module,
        "MANIFEST_FILE",
        processed / "manifest.jsonl",
    )

    stats = build_module.build_corpus()

    assert stats["total_files"] == 2
    assert stats["accepted"] == 1
    assert stats["content_duplicates"] == 1

    manifest = (processed / "manifest.jsonl").read_text(encoding="utf-8")
    records = [json.loads(line) for line in manifest.splitlines()]
    assert len(records) == 1
    assert records[0]["source"].endswith("first.txt")
