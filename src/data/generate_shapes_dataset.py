"""Utilities for procedurally generating a labeled geometric shapes dataset.

The module exposes :func:`generate_shapes_dataset` which can be called either from
Python or via the command line to produce a deterministic dataset of colored
geometric shapes. Images are rendered with Pillow and accompanied by structured
metadata describing the attributes used during generation.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
from PIL import Image, ImageDraw

__all__ = ["ShapeAttributes", "generate_shapes_dataset"]


@dataclass(frozen=True)
class ShapeAttributes:
    """Container describing a procedurally generated sample.

    Attributes
    ----------
    shape: str
        The name of the geometric primitive (e.g. ``"circle"``).
    color: str
        Human readable color label used when rendering the shape.
    size: str
        Relative size bucket (``"small"``, ``"medium"`` or ``"large"``).
    rotation: int
        Clockwise rotation in degrees applied when drawing the shape.
    split: str
        Name of the dataset split (``"train"`` or ``"test"``).
    seed: int
        Random seed used during creation which allows reproducing the image.
    filename: str
        Path (relative to the dataset root) of the rendered PNG file.
    """

    shape: str
    color: str
    size: str
    rotation: int
    split: str
    seed: int
    filename: str


def _shape_colors() -> Sequence[str]:
    return ("red", "green", "blue", "yellow")


def _shape_types() -> Sequence[str]:
    return ("circle", "square", "triangle", "pentagon")


def _size_buckets() -> Sequence[str]:
    return ("small", "medium", "large")


def _rotation_options() -> Sequence[int]:
    return (0, 30, 45, 60, 90)


def _color_to_rgb(color: str) -> tuple[int, int, int]:
    mapping = {
        "red": (220, 20, 60),
        "green": (34, 139, 34),
        "blue": (30, 144, 255),
        "yellow": (255, 215, 0),
    }
    return mapping[color]


def _draw_shape(draw: ImageDraw.ImageDraw, shape: str, bbox: Sequence[float], color: str) -> None:
    if shape == "circle":
        draw.ellipse(bbox, fill=color)
    elif shape == "square":
        draw.rectangle(bbox, fill=color)
    elif shape == "triangle":
        left, top, right, bottom = bbox
        draw.polygon([(left + right) / 2, top, left, bottom, right, bottom], fill=color)
    elif shape == "pentagon":
        left, top, right, bottom = bbox
        cx, cy = (left + right) / 2, (top + bottom) / 2
        radius = (right - left) / 2
        points: List[tuple[float, float]] = []
        for i in range(5):
            angle = math.radians(90 + i * 72)
            points.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle)))
        draw.polygon(points, fill=color)
    else:
        raise ValueError(f"Unsupported shape type: {shape}")


def _render_sample(
    shape: str,
    color: str,
    size: str,
    rotation: int,
    image_size: int = 128,
    rng: np.random.Generator | None = None,
) -> Image.Image:
    image = Image.new("RGBA", (image_size, image_size), color=(245, 245, 245, 255))
    draw = ImageDraw.Draw(image)

    padding = 16
    size_scale = {"small": 0.35, "medium": 0.55, "large": 0.75}[size]
    base_radius = (image_size / 2 - padding) * size_scale
    bbox = [image_size / 2 - base_radius, image_size / 2 - base_radius,
            image_size / 2 + base_radius, image_size / 2 + base_radius]

    temp = Image.new("RGBA", (image_size, image_size), (0, 0, 0, 0))
    temp_draw = ImageDraw.Draw(temp)
    _draw_shape(temp_draw, shape, bbox, _color_to_rgb(color))
    rotated = temp.rotate(rotation, resample=Image.BICUBIC, center=(image_size / 2, image_size / 2))
    image.alpha_composite(rotated)

    rng = rng or np.random.default_rng()
    noise = rng.normal(0, 3, (image_size, image_size, 3)).astype(np.int16)
    rgb_image = image.convert("RGB")
    noisy = np.clip(np.array(rgb_image).astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy, mode="RGB")


def _attribute_grid() -> Iterable[tuple[str, str, str, int]]:
    for shape in _shape_types():
        for color in _shape_colors():
            for size in _size_buckets():
                for rotation in _rotation_options():
                    yield shape, color, size, rotation


def generate_shapes_dataset(
    output_dir: str | Path = "data",
    image_size: int = 128,
    test_ratio: float = 0.2,
    seed: int | None = 1234,
) -> list[ShapeAttributes]:
    """Generate and persist a labeled dataset of geometric shapes.

    Parameters
    ----------
    output_dir:
        Directory where the ``images`` folder and ``metadata.json`` will be
        written. The directory is created if needed.
    image_size:
        Height and width (in pixels) of the generated images.
    test_ratio:
        Proportion of the dataset reserved for the ``test`` split.
    seed:
        Optional global random seed. Set ``None`` for nondeterministic output.

    Returns
    -------
    list[ShapeAttributes]
        Metadata for all generated samples.
    """

    rng = np.random.default_rng(seed)
    if seed is not None:
        random.seed(seed)

    output_path = Path(output_dir)
    images_path = output_path / "images"
    output_path.mkdir(parents=True, exist_ok=True)
    images_path.mkdir(parents=True, exist_ok=True)

    attributes: list[ShapeAttributes] = []
    combinations = list(_attribute_grid())
    random.shuffle(combinations)

    for idx, (shape, color, size, rotation) in enumerate(combinations):
        sample_seed = int(rng.integers(0, 2**32 - 1))
        image_rng = np.random.default_rng(sample_seed)
        image = _render_sample(shape, color, size, rotation, image_size=image_size, rng=image_rng)
        split = "test" if idx < max(1, int(len(combinations) * test_ratio)) else "train"
        filename = f"images/{idx:04d}_{shape}_{color}_{size}_{rotation}.png"
        image.save(output_path / filename)

        attributes.append(
            ShapeAttributes(
                shape=shape,
                color=color,
                size=size,
                rotation=rotation,
                split=split,
                seed=sample_seed,
                filename=filename,
            )
        )

    metadata_path = output_path / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump([asdict(attr) for attr in attributes], f, indent=2)

    return attributes


def main(args: Sequence[str] | None = None) -> None:
    """Command line entry point for generating the dataset."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate a procedural shapes dataset")
    parser.add_argument("--output-dir", type=Path, default=Path("data"), help="Directory where the dataset will be stored")
    parser.add_argument("--image-size", type=int, default=128, help="Square image dimension in pixels")
    parser.add_argument("--test-ratio", type=float, default=0.2, help="Fraction reserved for the test split")
    parser.add_argument("--seed", type=int, default=1234, help="Global random seed for reproducibility")

    parsed = parser.parse_args(args=args)
    generate_shapes_dataset(
        output_dir=parsed.output_dir,
        image_size=parsed.image_size,
        test_ratio=parsed.test_ratio,
        seed=parsed.seed,
    )


if __name__ == "__main__":
    main()
