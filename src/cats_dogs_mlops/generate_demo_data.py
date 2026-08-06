from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw


def generate_cat(path: Path, rng: random.Random, size: int) -> None:
    image = Image.new("RGB", (size, size), (235, 225, 205))
    draw = ImageDraw.Draw(image)
    cx, cy = size // 2 + rng.randint(-8, 8), size // 2 + rng.randint(-8, 8)
    radius = size // 4
    fur = (rng.randint(160, 230), rng.randint(90, 160), rng.randint(40, 100))
    draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=fur)
    draw.polygon([(cx - radius, cy - radius // 2), (cx - radius // 2, cy - radius * 2), (cx, cy - radius)], fill=fur)
    draw.polygon([(cx + radius, cy - radius // 2), (cx + radius // 2, cy - radius * 2), (cx, cy - radius)], fill=fur)
    draw.ellipse((cx - radius // 2, cy - 8, cx - radius // 4, cy + 6), fill=(30, 120, 40))
    draw.ellipse((cx + radius // 4, cy - 8, cx + radius // 2, cy + 6), fill=(30, 120, 40))
    draw.polygon([(cx, cy + 6), (cx - 7, cy + 16), (cx + 7, cy + 16)], fill=(190, 70, 90))
    image.save(path, quality=90)


def generate_dog(path: Path, rng: random.Random, size: int) -> None:
    image = Image.new("RGB", (size, size), (205, 225, 235))
    draw = ImageDraw.Draw(image)
    cx, cy = size // 2 + rng.randint(-8, 8), size // 2 + rng.randint(-8, 8)
    radius = size // 4
    fur = (rng.randint(80, 150), rng.randint(50, 110), rng.randint(25, 70))
    draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=fur)
    draw.ellipse((cx - radius * 2, cy - radius, cx - radius // 2, cy + radius), fill=fur)
    draw.ellipse((cx + radius // 2, cy - radius, cx + radius * 2, cy + radius), fill=fur)
    draw.ellipse((cx - radius // 2, cy - 8, cx - radius // 4, cy + 6), fill=(25, 25, 25))
    draw.ellipse((cx + radius // 4, cy - 8, cx + radius // 2, cy + 6), fill=(25, 25, 25))
    draw.ellipse((cx - 10, cy + 3, cx + 10, cy + 20), fill=(25, 25, 25))
    draw.arc((cx - 22, cy + 8, cx + 22, cy + 38), start=10, end=170, fill=(240, 170, 170), width=4)
    image.save(path, quality=90)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic demo images for offline pipeline validation.")
    parser.add_argument("--output", default="data/raw/demo")
    parser.add_argument("--per-class", type=int, default=40)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    root = Path(args.output)
    rng = random.Random(args.seed)
    for label in ("cats", "dogs"):
        (root / label).mkdir(parents=True, exist_ok=True)
    for index in range(args.per_class):
        generate_cat(root / "cats" / f"cat.{index}.jpg", rng, args.size)
        generate_dog(root / "dogs" / f"dog.{index}.jpg", rng, args.size)
    print(f"Generated {args.per_class * 2} demo images in {root}")


if __name__ == "__main__":
    main()
