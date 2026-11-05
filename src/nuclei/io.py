"""Finding and loading BBBC039 images, ground-truth masks and splits.

Everything else in the project goes through these functions, so if a file
format surprise turns up, it gets fixed in one place.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import tifffile
from skimage.measure import label

# The metadata package lists the image names belonging to each split.
SPLIT_FILES = {
    "training": "training.txt",
    "validation": "validation.txt",
    "test": "test.txt",
}


@dataclass
class Sample:
    """One field of view: its image, its mask (if found) and its split."""

    name: str              # file stem shared by the image and its mask
    split: str             # "training", "validation", "test" or "unassigned"
    image_path: Path
    mask_path: Path | None


def _real_files(root: Path, patterns: list[str]) -> list[Path]:
    """Recursively find files, skipping macOS zip debris (__MACOSX, ._*)."""
    found = []
    for pattern in patterns:
        for p in root.rglob(pattern):
            if "__MACOSX" in p.parts or p.name.startswith("._"):
                continue
            found.append(p)
    return sorted(found)


def read_splits(root: Path) -> dict[str, str]:
    """Map each image stem to its split, using training/validation/test.txt.

    Raises an error if one image is listed in two splits: that would leak
    test information into training, and we want to know immediately.
    """
    stem_to_split: dict[str, str] = {}
    for split, filename in SPLIT_FILES.items():
        matches = _real_files(root, [filename])
        if not matches:
            print(f"WARNING: {filename} not found under {root}")
            continue
        for line in matches[0].read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            stem = Path(line).stem  # works whether the list has .png, .tif or no extension
            if stem in stem_to_split and stem_to_split[stem] != split:
                raise ValueError(
                    f"{stem} is listed in both {stem_to_split[stem]} and {split}"
                )
            stem_to_split[stem] = split
    return stem_to_split


def find_samples(root: Path) -> list[Sample]:
    """Pair every image with its mask and split."""
    root = Path(root)
    images = _real_files(root, ["*.tif", "*.tiff"])
    masks = {p.stem: p for p in _real_files(root, ["*.png"])}
    splits = read_splits(root)
    return [
        Sample(
            name=img.stem,
            split=splits.get(img.stem, "unassigned"),
            image_path=img,
            mask_path=masks.get(img.stem),
        )
        for img in images
    ]


def load_image(path: Path) -> np.ndarray:
    """Read a microscopy image, keeping its original dtype (16-bit here).

    We never convert to 8-bit on load: that would throw away intensity
    resolution before we know whether we need it.
    """
    img = tifffile.imread(path)
    if img.ndim != 2:
        raise ValueError(f"Expected a single-channel 2D image, got shape {img.shape}: {path}")
    return img


def read_mask_raw(path: Path) -> np.ndarray:
    """Read a ground-truth PNG exactly as stored (usually H x W x 3 colours)."""
    return iio.imread(path)


def _colour_code(raw: np.ndarray) -> np.ndarray:
    """Turn each pixel's colour into a single integer, 0 for black background."""
    if raw.ndim == 2:
        return raw.astype(np.int64)
    rgb = raw[..., :3].astype(np.int64)  # drop an alpha channel if present
    return (rgb[..., 0] << 16) | (rgb[..., 1] << 8) | rgb[..., 2]


def decode_mask(raw: np.ndarray) -> np.ndarray:
    """Convert a BBBC039 colour-coded mask into a label image.

    In the PNGs, each nucleus is a patch of one colour, and touching nuclei
    get different colours. `label` numbers connected regions of *equal value*,
    so after collapsing RGB into one integer per pixel, each nucleus becomes
    its own label (1, 2, 3, ...), with 0 as background.
    """
    return label(_colour_code(raw), background=0)


def decode_mask_first_channel(raw: np.ndarray) -> np.ndarray:
    """The decoding from the dataset authors' example: first channel only.

    Kept only as a cross-check. If two touching nuclei differ in green or
    blue but share the same red value, this version merges them.
    """
    first = raw if raw.ndim == 2 else raw[..., 0]
    return label(first.astype(np.int64), background=0)


def load_mask(path: Path) -> np.ndarray:
    """Read and decode a ground-truth mask in one step."""
    return decode_mask(read_mask_raw(path))
