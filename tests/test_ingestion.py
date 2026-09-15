from data.ingestion import iter_supported_files, iter_text_files


def test_iter_supported_files_is_recursive(tmp_path):
    nested = tmp_path / "nested"
    nested.mkdir()
    (tmp_path / "top.txt").write_text("top", encoding="utf-8")
    (nested / "paper.md").write_text("paper", encoding="utf-8")
    (nested / "paper.pdf").write_bytes(b"pdf")
    (nested / "ignore.csv").write_text("ignore", encoding="utf-8")

    paths = list(iter_supported_files(tmp_path))

    assert [path.name for path in paths] == ["paper.md", "paper.pdf", "top.txt"]


def test_iter_text_files_reads_nested_text(tmp_path):
    nested = tmp_path / "nested"
    nested.mkdir()
    source = nested / "paper.txt"
    source.write_text("medical text", encoding="utf-8")

    result = list(iter_text_files(tmp_path))

    assert result == [(str(source), "medical text")]
