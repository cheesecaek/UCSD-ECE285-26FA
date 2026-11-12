"""Dataset helpers for the directory-per-style WikiArt layout."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Callable

from PIL import Image, ImageFile
from torch import Tensor
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class WikiArtDataset(Dataset[Tensor]):
    """Treat every image below ``root`` as an unlabeled training example."""

    def __init__(
        self,
        root: str | Path,
        transform: Callable,
        max_images: int | None = None,
        seed: int = 285,
    ) -> None:
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"Dataset directory does not exist: {self.root}")
        self.paths = sorted(
            path for path in self.root.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self.paths:
            raise RuntimeError(f"No supported images found below {self.root}")
        if max_images is not None and max_images < len(self.paths):
            rng = random.Random(seed)
            self.paths = rng.sample(self.paths, max_images)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Tensor:
        # A few public WikiArt mirrors contain truncated files. Pillow can read
        # most of them; if one is irrecoverable, use the next image rather than
        # losing an entire multi-hour run.
        last_error: Exception | None = None
        for offset in range(min(10, len(self.paths))):
            path = self.paths[(index + offset) % len(self.paths)]
            try:
                with Image.open(path) as image:
                    return self.transform(image.convert("RGB"))
            except (OSError, ValueError) as error:
                last_error = error
        raise RuntimeError(f"Could not decode an image near index {index}") from last_error
