import json

from tokenizer.train_tokenizer_resumable import (
    add_pair_delta,
    merge_pair,
    open_db,
    pair_counts,
)


def test_merge_pair_merges_non_overlapping_occurrences():
    assert merge_pair(list("aaaa"), "a", "a") == ["aa", "aa"]
    assert merge_pair(list("ababa"), "a", "b") == ["ab", "ab", "a"]


def test_pair_counts_are_frequency_weighted_and_checkpointed(tmp_path):
    db = open_db(tmp_path / "state.sqlite")
    try:
        with db:
            db.execute(
                "INSERT INTO pieces(piece,freq,symbols) VALUES(?,?,?)",
                ("abab", 3, json.dumps(list("abab"))),
            )
            for (left, right), occurrences in pair_counts(list("abab")).items():
                db.execute(
                    "INSERT INTO piece_pairs(piece_id,left_symbol,right_symbol,occurrences) "
                    "VALUES(1,?,?,?)",
                    (left, right, occurrences),
                )
                db.execute(
                    "INSERT INTO pair_counts(left_symbol,right_symbol,count) VALUES(?,?,?)",
                    (left, right, occurrences * 3),
                )

        assert db.execute(
            "SELECT count FROM pair_counts WHERE left_symbol='a' AND right_symbol='b'"
        ).fetchone()[0] == 6

        # Reopening the SQLite file must preserve the durable state.
        db.close()
        db = open_db(tmp_path / "state.sqlite")
        assert db.execute("SELECT freq FROM pieces WHERE piece='abab'").fetchone()[0] == 3
        assert db.execute(
            "SELECT count FROM pair_counts WHERE left_symbol='b' AND right_symbol='a'"
        ).fetchone()[0] == 3
    finally:
        db.close()


def test_pair_delta_removes_zero_count_pair(tmp_path):
    db = open_db(tmp_path / "state.sqlite")
    try:
        with db:
            db.execute(
                "INSERT INTO pair_counts(left_symbol,right_symbol,count) VALUES('a','b',2)"
            )
            add_pair_delta(db, "a", "b", -2)
        assert db.execute(
            "SELECT 1 FROM pair_counts WHERE left_symbol='a' AND right_symbol='b'"
        ).fetchone() is None
    finally:
        db.close()
