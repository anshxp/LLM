from collections import defaultdict
from pathlib import Path

from data.file_hash import calculate_sha256


def find_duplicates(data_dir: str | Path) -> dict[str, list[Path]]:
    """Find exact duplicate files using SHA-256."""
    data_dir = Path(data_dir)

    hashes = defaultdict(list)

    for path in sorted(data_dir.rglob("*.pdf")):
        file_hash = calculate_sha256(path)
        hashes[file_hash].append(path)

    return {
        file_hash: paths
        for file_hash, paths in hashes.items()
        if len(paths) > 1
    }


if __name__ == "__main__":
    duplicates = find_duplicates("data/raw")

    if not duplicates:
        print("No exact duplicate PDFs found.")
    else:
        print(f"Exact duplicate groups: {len(duplicates)}")
        print()

        for index, paths in enumerate(duplicates.values(), start=1):
            print(f"Duplicate group {index}:")
            for path in paths:
                print(f"  {path}")
            print()