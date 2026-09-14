from data.inventory import inventory_directory


def test_inventory_counts_supported_files_recursively(tmp_path):
    nested = tmp_path / "nested"
    nested.mkdir()
    (tmp_path / "a.txt").write_text("abc", encoding="utf-8")
    (nested / "b.md").write_text("12345", encoding="utf-8")
    (nested / "ignore.csv").write_text("ignored", encoding="utf-8")

    result = inventory_directory(tmp_path)

    assert result["total_files"] == 2
    assert result["total_size_bytes"] == 8
    assert result["extensions"] == {".md": 1, ".txt": 1}
