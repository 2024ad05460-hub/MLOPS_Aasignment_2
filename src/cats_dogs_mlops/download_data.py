from __future__ import annotations

import argparse
import shutil
import subprocess
import zipfile
from pathlib import Path


def run(command: list[str]) -> None:
    print("+", " ".join(command))
    subprocess.run(command, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the Kaggle Cats vs Dogs dataset using the Kaggle CLI.")
    parser.add_argument("--destination", default="data/raw")
    parser.add_argument("--mode", choices=["competition", "dataset"], default="competition")
    parser.add_argument("--slug", default="dogs-vs-cats", help="Competition or dataset slug.")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    destination = Path(args.destination)
    if destination.exists() and any(destination.iterdir()):
        if not args.force:
            raise SystemExit(f"{destination} is not empty. Use --force to replace it.")
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)

    if shutil.which("kaggle") is None:
        raise SystemExit("Kaggle CLI not found. Install requirements.txt and configure ~/.kaggle/kaggle.json.")

    archive = destination / "download.zip"
    if args.mode == "competition":
        run(["kaggle", "competitions", "download", "-c", args.slug, "-p", str(destination)])
        candidates = list(destination.glob("*.zip"))
    else:
        run(["kaggle", "datasets", "download", "-d", args.slug, "-p", str(destination)])
        candidates = list(destination.glob("*.zip"))
    if not candidates:
        raise SystemExit("Kaggle CLI completed but no zip archive was found")
    archive = candidates[0]
    with zipfile.ZipFile(archive) as handle:
        handle.extractall(destination)
    archive.unlink(missing_ok=True)
    print(f"Dataset extracted to {destination.resolve()}")


if __name__ == "__main__":
    main()
