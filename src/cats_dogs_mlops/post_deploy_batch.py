from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Send a balanced batch of labeled test images to the deployed "
            "API and record feedback for post-deployment accuracy."
        )
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="Base URL of the deployed FastAPI service.",
    )
    parser.add_argument(
        "--manifest",
        default="data/processed/manifest.csv",
        help="Path to the processed dataset manifest.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Total number of samples. Use an even number for a balanced batch.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.limit <= 0:
        raise ValueError("--limit must be greater than zero.")

    if args.limit % 2 != 0:
        raise ValueError(
            "--limit must be an even number so the batch can contain "
            "equal cat and dog samples."
        )

    manifest = Path(args.manifest)

    if not manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest}")

    root = manifest.parent

    with manifest.open(newline="", encoding="utf-8") as handle:
        test_rows = [
            row
            for row in csv.DictReader(handle)
            if row.get("split", "").strip().lower() == "test"
        ]

    per_class = args.limit // 2

    cat_rows = [
        row
        for row in test_rows
        if row.get("label_name", "").strip().lower() == "cat"
    ][:per_class]

    dog_rows = [
        row
        for row in test_rows
        if row.get("label_name", "").strip().lower() == "dog"
    ][:per_class]

    rows = cat_rows + dog_rows

    if len(cat_rows) < per_class or len(dog_rows) < per_class:
        raise RuntimeError(
            f"Insufficient balanced test samples. "
            f"Required {per_class} cats and {per_class} dogs, "
            f"but found {len(cat_rows)} cats and {len(dog_rows)} dogs."
        )

    outcomes: list[dict[str, object]] = []

    for row in rows:
        image_path = root / row["relative_path"]
        true_label = row["label_name"].strip().lower()

        if not image_path.exists():
            raise FileNotFoundError(
                f"Manifest image does not exist: {image_path}"
            )

        with image_path.open("rb") as image_handle:
            response = requests.post(
                f"{args.base_url}/predict",
                files={
                    "file": (
                        image_path.name,
                        image_handle,
                        "image/jpeg",
                    )
                },
                timeout=30,
            )

        response.raise_for_status()
        prediction = response.json()

        feedback = requests.post(
            f"{args.base_url}/feedback",
            json={
                "request_id": prediction["request_id"],
                "true_label": true_label,
            },
            timeout=15,
        )
        feedback.raise_for_status()

        predicted_label = prediction["label"].strip().lower()

        outcomes.append(
            {
                "file": row["relative_path"],
                "predicted": predicted_label,
                "true": true_label,
                "confidence": prediction.get("confidence"),
                "latency_ms": prediction.get("latency_ms"),
                "correct": predicted_label == true_label,
            }
        )

    performance = requests.get(
        f"{args.base_url}/monitoring/performance",
        timeout=15,
    )
    performance.raise_for_status()

    batch_correct = sum(
        1 for outcome in outcomes if outcome["correct"] is True
    )
    batch_accuracy = batch_correct / len(outcomes)

    result = {
        "batch_summary": {
            "sample_count": len(outcomes),
            "cat_count": len(cat_rows),
            "dog_count": len(dog_rows),
            "correct_count": batch_correct,
            "accuracy": batch_accuracy,
        },
        "requests": outcomes,
        "cumulative_performance": performance.json(),
    }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()