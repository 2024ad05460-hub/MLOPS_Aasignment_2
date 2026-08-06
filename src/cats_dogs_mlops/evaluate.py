from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

from .data import ManifestDataset
from .model import load_model_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the serialized model on the held-out test split.")
    parser.add_argument("--manifest", default="data/processed/manifest.csv")
    parser.add_argument("--model", default="models/model.pt")
    parser.add_argument("--metrics-out", default="reports/test_metrics.json")
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = load_model_bundle(args.model)
    dataset = ManifestDataset(args.manifest, split="test", image_size=bundle.image_size, augment=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    truth: list[int] = []
    predictions: list[int] = []
    with torch.inference_mode():
        for images, labels in loader:
            logits = bundle.model(images)
            predictions.extend(torch.argmax(logits, dim=1).tolist())
            truth.extend(labels.tolist())
    metrics = {
        "accuracy": float(accuracy_score(truth, predictions)),
        "precision_macro": float(precision_score(truth, predictions, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(truth, predictions, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(truth, predictions, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(truth, predictions, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            truth, predictions, target_names=bundle.class_names, output_dict=True, zero_division=0
        ),
        "test_samples": len(dataset),
    }
    output = Path(args.metrics_out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
