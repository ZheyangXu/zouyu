import shutil
from pathlib import Path


def update_assets(assets: dict[str, bytes], source_dir: Path, meshdir: str) -> None:
    if not source_dir.is_dir():
        return
    for path in source_dir.rglob("*"):
        if path.is_file():
            rel = path.relative_to(source_dir)
            dest = Path(meshdir) / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            assets[str(rel)] = path.read_bytes()
            if not dest.exists():
                shutil.copy2(path, dest)
