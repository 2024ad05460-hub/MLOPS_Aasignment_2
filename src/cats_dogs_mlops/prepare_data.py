from __future__ import annotations

import argparse
import json

from .data import prepare_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare cats-vs-dogs data into stratified train/val/test folders.")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-per-class", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = prepare_dataset(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        image_size=args.image_size,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        max_per_class=args.max_per_class,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
