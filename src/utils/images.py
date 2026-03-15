from pathlib import Path
from typing import Iterable

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def get_image_files(
    data_dir: Path,
    extensions: Iterable[str] = IMAGE_EXTENSIONS,
) -> list[Path]:
    exts = {ext.lower() for ext in extensions}

    return [
        path
        for path in data_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in exts
    ]
