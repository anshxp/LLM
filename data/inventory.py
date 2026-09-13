from pathlib import Path


def inventory_directory(data_dir: str | Path) -> dict:
    """
    Inventory files in a directory without reading their contents.
    """
    data_dir = Path(data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"Directory does not exist: {data_dir}")

    files = [
        path
        for path in data_dir.rglob("*")
        if path.is_file()
    ]

    extensions = {}

    for path in files:
        extension = path.suffix.lower() or "[no extension]"
        extensions[extension] = extensions.get(extension, 0) + 1

    total_size = sum(path.stat().st_size for path in files)

    return {
        "total_files": len(files),
        "total_size_bytes": total_size,
        "extensions": extensions,
        "files": files,
    }


def print_inventory(data_dir: str | Path) -> None:
    """Print a human-readable inventory."""
    inventory = inventory_directory(data_dir)

    total_size_mb = inventory["total_size_bytes"] / (1024 ** 2)

    print(f"Directory: {data_dir}")
    print(f"Total files: {inventory['total_files']}")
    print(f"Total size: {total_size_mb:.2f} MB")

    print("\nFile types:")

    for extension, count in sorted(
        inventory["extensions"].items()
    ):
        print(f"  {extension}: {count}")

    print("\nFiles:")

    for path in inventory["files"]:
        size_mb = path.stat().st_size / (1024 ** 2)
        print(f"  {path.name} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    print_inventory("data/raw")