from pathlib import Path

def print_tree(path: Path, prefix: str = ""):
    items = list(path.iterdir())
    for i, item in enumerate(items):
        connector = "├── " if i < len(items) - 1 else "└── "
        print(prefix + connector + item.name)
        if item.is_dir():
            extension = "│   " if i < len(items) - 1 else "    "
            print_tree(item, prefix + extension)

print_tree(Path("."))
