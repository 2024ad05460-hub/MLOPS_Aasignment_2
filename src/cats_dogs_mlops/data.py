from __future__ import annotations

import csv
import hashlib
import json
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageOps, UnidentifiedImageError
import torch
from torch.utils.data import Dataset
from torchvision import transforms


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def preprocess_image(image: Image.Image, image_size: int = 224) -> torch.Tensor:
    """Convert any PIL image to normalized 3xHxW RGB tensor."""
    pipeline = transforms.Compose(
        [
            transforms.Lambda(lambda img: ImageOps.exif_transpose(img).convert("RGB")),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return pipeline(image)


def train_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Lambda(lambda img: ImageOps.exif_transpose(img).convert("RGB")),
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(12),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.12),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def eval_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Lambda(lambda img: ImageOps.exif_transpose(img).convert("RGB")),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


class ManifestDataset(Dataset):
    def __init__(self, manifest_path: str | Path, split: str, image_size: int = 224, augment: bool = False) -> None:
        self.manifest_path = Path(manifest_path)
        self.root = self.manifest_path.parent
        with self.manifest_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.rows = [row for row in rows if row["split"] == split]
        if not self.rows:
            raise ValueError(f"No rows for split '{split}' in {self.manifest_path}")
        self.transform = train_transform(image_size) if augment else eval_transform(image_size)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.rows[index]
        path = self.root / row["relative_path"]
        with Image.open(path) as image:
            tensor = self.transform(image)
        return tensor, int(row["label"])


def _infer_label(path: Path) -> str | None:
    lower_parts = [part.lower() for part in path.parts]
    name = path.name.lower()
    if "cat" in lower_parts or name.startswith("cat") or ".cat." in name:
        return "cat"
    if "cats" in lower_parts:
        return "cat"
    if "dog" in lower_parts or name.startswith("dog") or ".dog." in name:
        return "dog"
    if "dogs" in lower_parts:
        return "dog"
    return None


def discover_images(raw_dir: str | Path) -> list[tuple[Path, str]]:
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory does not exist: {raw_dir}")
    samples: list[tuple[Path, str]] = []
    for path in raw_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS:
            label = _infer_label(path)
            if label:
                samples.append((path, label))
    if not samples:
        raise ValueError(
            "No labeled cat/dog images found. Expected class folders (cats/dogs) or filenames such as cat.1.jpg and dog.1.jpg."
        )
    return samples


def _split_class(paths: list[Path], train_ratio: float, val_ratio: float, seed: int) -> dict[str, list[Path]]:
    rng = random.Random(seed)
    paths = list(paths)
    rng.shuffle(paths)
    n = len(paths)
    n_train = max(1, int(n * train_ratio))
    n_val = max(1, int(n * val_ratio))
    if n_train + n_val >= n:
        n_train = max(1, n - 2)
        n_val = 1
    return {
        "train": paths[:n_train],
        "val": paths[n_train : n_train + n_val],
        "test": paths[n_train + n_val :],
    }


def prepare_dataset(
    raw_dir: str | Path,
    processed_dir: str | Path,
    image_size: int = 224,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
    max_per_class: int | None = None,
) -> dict[str, object]:
    """Validate, resize, stratify, and materialize an 80/10/10 dataset."""
    if not 0 < train_ratio < 1 or not 0 < val_ratio < 1 or train_ratio + val_ratio >= 1:
        raise ValueError("Ratios must be positive and train_ratio + val_ratio < 1")

    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    if processed_dir.exists():
        shutil.rmtree(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    discovered = discover_images(raw_dir)
    grouped: dict[str, list[Path]] = {"cat": [], "dog": []}
    for path, label in discovered:
        grouped[label].append(path)

    if max_per_class:
        for label in grouped:
            grouped[label] = sorted(grouped[label])[:max_per_class]

    rows: list[dict[str, str | int]] = []
    rejected: list[str] = []
    label_to_id = {"cat": 0, "dog": 1}

    for label, paths in grouped.items():
        if len(paths) < 3:
            raise ValueError(f"Need at least 3 valid images for class '{label}', found {len(paths)}")
        split_map = _split_class(paths, train_ratio, val_ratio, seed + label_to_id[label])
        for split, split_paths in split_map.items():
            for source in split_paths:
                try:
                    with Image.open(source) as image:
                        image = ImageOps.exif_transpose(image).convert("RGB")
                        image = image.resize((image_size, image_size), Image.Resampling.LANCZOS)
                        digest = hashlib.sha256(source.read_bytes()).hexdigest()[:12]
                        destination = processed_dir / split / label / f"{source.stem}_{digest}.jpg"
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        image.save(destination, format="JPEG", quality=92, optimize=True)
                except (UnidentifiedImageError, OSError, ValueError) as exc:
                    rejected.append(f"{source}: {exc}")
                    continue
                rows.append(
                    {
                        "relative_path": destination.relative_to(processed_dir).as_posix(),
                        "split": split,
                        "label_name": label,
                        "label": label_to_id[label],
                        "source_path": str(source),
                    }
                )

    manifest = processed_dir / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    counts = Counter((row["split"], row["label_name"]) for row in rows)
    summary = {
        "image_size": image_size,
        "seed": seed,
        "ratios": {"train": train_ratio, "validation": val_ratio, "test": 1 - train_ratio - val_ratio},
        "total_valid_images": len(rows),
        "rejected_images": rejected,
        "counts": {f"{split}_{label}": count for (split, label), count in sorted(counts.items())},
        "class_mapping": label_to_id,
    }
    (processed_dir / "data_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def iter_manifest_rows(manifest_path: str | Path, split: str | None = None) -> Iterable[dict[str, str]]:
    with Path(manifest_path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if split is None or row["split"] == split:
                yield row
