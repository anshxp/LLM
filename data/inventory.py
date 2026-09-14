from pathlib import Path

from data.ingestion import SUPPORTED_EXTENSIONS


def inventory_directory(data_dir: str | Path) -> dict:
    """Inventory supported source files recursively without reading contents."""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Directory does not exist: {data_dir}")
    if not data_dir.is_dir():
        raise NotADirectoryError(f"Expected a directory: {data_dir}")

    files = [
        path
        for path in data_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    extensions: dict[str, int] = {}
    total_size = 0
    for path in files:
        extension = path.suffix.lower()
        extensions[extension] = extensions.get(extension, 0) + 1
        total_size += path.stat().st_size

    return {
        "total_files": len(files),
        "total_size_bytes": total_size,
        "extensions": dict(sorted(extensions.items())),
        "files": sorted(files),
    }


def print_inventory(data_dir: str | Path) -> None:
    """Print a human-readable inventory of supported source files."""
    inventory = inventory_directory(data_dir)
    total_size_mb = inventory["total_size_bytes"] / (1024**2)

    print(f"Directory: {data_dir}")
    print(f"Supported files: {inventory['total_files']}")
    print(f"Total size: {total_size_mb:.2f} MB")
    print("\nFile types:")
    for extension, count in inventory["extensions"].items():
        print(f"  {extension}: {count}")


if __name__ == "__main__":
    print_inventory("data/raw")
