from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send labeled test images and record feedback for post-deployment accuracy.")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--manifest", default="data/processed/manifest.csv")
    parser.add_argument("--limit", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    import csv

    args = parse_args()
    manifest = Path(args.manifest)
    root = manifest.parent
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] == "test"][: args.limit]
    outcomes = []
    for row in rows:
        image_path = root / row["relative_path"]
        with image_path.open("rb") as handle:
            response = requests.post(
                f"{args.base_url}/predict",
                files={"file": (image_path.name, handle, "image/jpeg")},
                timeout=30,
            )
        response.raise_for_status()
        prediction = response.json()
        feedback = requests.post(
            f"{args.base_url}/feedback",
            json={"request_id": prediction["request_id"], "true_label": row["label_name"]},
            timeout=15,
        )
        feedback.raise_for_status()
        outcomes.append(
            {
                "file": row["relative_path"],
                "predicted": prediction["label"],
                "true": row["label_name"],
                "correct": prediction["label"] == row["label_name"],
            }
        )
    performance = requests.get(f"{args.base_url}/monitoring/performance", timeout=15)
    performance.raise_for_status()
    print(json.dumps({"requests": outcomes, "performance": performance.json()}, indent=2))


if __name__ == "__main__":
    main()
