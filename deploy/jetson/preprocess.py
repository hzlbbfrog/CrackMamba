"""Shared image preprocessing and probability export for Jetson inference."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_SIZE = (480, 480)


def collect_images(source: str | Path) -> list[Path]:
    """Return one supported image or recursively sorted images from a directory."""
    source = Path(source).expanduser().resolve()
    if source.is_file():
        paths = [source] if source.suffix.lower() in IMAGE_SUFFIXES else []
    elif source.is_dir():
        paths = [path for path in source.rglob("*") if path.is_file()
                 and path.suffix.lower() in IMAGE_SUFFIXES]
    else:
        raise FileNotFoundError(f"Image source does not exist: {source}")
    paths.sort(key=lambda path: (path.as_posix().casefold(), path.as_posix()))
    if not paths:
        raise ValueError(f"No supported images found in {source}")
    return paths


def load_image(path: str | Path, size: tuple[int, int] = DEFAULT_SIZE,
               resize: bool = False) -> np.ndarray:
    """Convert RGB pixels to contiguous float32 NCHW in [-1, 1].

    ``size`` is (height, width). Images are never resized implicitly. Explicit
    resizing uses Pillow bilinear interpolation and changes the input geometry.
    """
    height, width = size
    if height <= 0 or width <= 0:
        raise ValueError("Image dimensions must be positive")
    with Image.open(path) as image:
        image = image.convert("RGB")
        if image.size != (width, height):
            if not resize:
                raise ValueError(f"{path}: expected {width}x{height} pixels, got "
                                 f"{image.width}x{image.height}; use --resize explicitly")
            image = image.resize((width, height), Image.Resampling.BILINEAR)
        pixels = np.asarray(image, dtype=np.float32)
    pixels = (pixels / np.float32(255.0) - np.float32(0.5)) / np.float32(0.5)
    return np.ascontiguousarray(pixels.transpose(2, 0, 1)[None])


def validate_output_directory(image_source: str | Path, output_dir: str | Path) -> None:
    """Keep recursive image discovery separate from generated PNG predictions."""
    source = Path(image_source).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if source.is_dir() and (output == source or source in output.parents):
        raise ValueError("The output directory must be outside the input image directory")


def probability_map(probability: np.ndarray) -> np.ndarray:
    """Validate a single foreground probability map without applying sigmoid."""
    array = np.asarray(probability)
    if array.ndim == 4 and array.shape[:2] == (1, 1):
        array = array[0, 0]
    elif array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2 or not array.size:
        raise ValueError(f"Expected one nonempty probability map, got {array.shape}")
    array = np.asarray(array, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError("Prediction contains NaN or infinite values")
    if array.min() < -1e-4 or array.max() > 1.0001:
        raise ValueError("Expected probabilities in [0, 1]; logits are not supported")
    return np.clip(array, 0.0, 1.0)


def save_prediction(probability: np.ndarray, image: str | Path,
                    image_source: str | Path, output_dir: str | Path,
                    threshold: float | None = None) -> dict[str, str]:
    """Save float probabilities, an 8-bit preview and an optional binary mask.

    Relative input directories and the complete image filename are retained so
    inputs such as ``a.png`` and ``a.jpg`` cannot overwrite each other's outputs.
    """
    array = probability_map(probability)
    if threshold is not None and not 0.0 <= threshold <= 1.0:
        raise ValueError("The binary threshold must be in [0, 1]")
    source = Path(image_source).expanduser().resolve()
    image = Path(image).expanduser().resolve()
    relative = image.relative_to(source if source.is_dir() else source.parent)
    destination = Path(output_dir) / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    paths = {
        "probability": str(destination) + ".probability.npy",
        "preview": str(destination) + ".probability.png",
    }
    np.save(paths["probability"], array, allow_pickle=False)
    Image.fromarray(np.rint(array * 255.0).astype(np.uint8)).save(paths["preview"])
    if threshold is not None:
        paths["binary"] = str(destination) + ".binary.png"
        Image.fromarray((array >= threshold).astype(np.uint8) * 255).save(paths["binary"])
    return paths


def sha256_file(path: str | Path) -> str:
    """Hash a model artifact without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_report(report: dict, destination: str | Path) -> None:
    """Write a UTF-8 JSON report with deterministic key ordering."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
