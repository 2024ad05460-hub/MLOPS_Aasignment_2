from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader

from .data import ManifestDataset
from .model import build_model, save_model_bundle


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> tuple[float, list[int], list[int]]:
    model.eval()
    losses: list[float] = []
    truth: list[int] = []
    preds: list[int] = []
    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            losses.append(float(loss.item()))
            preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
            truth.extend(labels.cpu().tolist())
    return float(np.mean(losses)), truth, preds


def save_curves(history: dict[str, list[float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_confusion_matrix(matrix: np.ndarray, class_names: list[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 4))
    plt.imshow(matrix)
    plt.title("Validation confusion matrix")
    plt.colorbar()
    plt.xticks(range(len(class_names)), class_names)
    plt.yticks(range(len(class_names)), class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            plt.text(j, i, str(matrix[i, j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train baseline or transfer-learning cats-vs-dogs model.")
    parser.add_argument("--manifest", default="data/processed/manifest.csv")
    parser.add_argument("--model-out", default="models/model.pt")
    parser.add_argument("--metrics-out", default="reports/train_metrics.json")
    parser.add_argument("--figures-dir", default="figures")
    parser.add_argument("--architecture", choices=["tinycnn", "mobilenet_v3_small"], default="tinycnn")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--experiment-name", default="cats-dogs-classification")
    parser.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = ["cat", "dog"]

    train_dataset = ManifestDataset(args.manifest, split="train", image_size=args.image_size, augment=True)
    val_dataset = ManifestDataset(args.manifest, split="val", image_size=args.image_size, augment=False)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = build_model(args.architecture, num_classes=2, pretrained=args.pretrained)
    if args.architecture == "mobilenet_v3_small" and args.freeze_backbone:
        for parameter in model.features.parameters():
            parameter.requires_grad = False
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)
    figures_dir = Path(args.figures_dir)
    metrics_out = Path(args.metrics_out)
    metrics_out.parent.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run() as run:
        mlflow.log_params(
            {
                "architecture": args.architecture,
                "pretrained": args.pretrained,
                "freeze_backbone": args.freeze_backbone,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "image_size": args.image_size,
                "seed": args.seed,
                "train_samples": len(train_dataset),
                "validation_samples": len(val_dataset),
                "device": str(device),
            }
        )
        history = {"train_loss": [], "val_loss": [], "val_accuracy": []}
        best_accuracy = -1.0
        best_state: dict[str, torch.Tensor] | None = None
        started = time.perf_counter()

        for epoch in range(args.epochs):
            model.train()
            batch_losses: list[float] = []
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                batch_losses.append(float(loss.item()))

            train_loss = float(np.mean(batch_losses))
            val_loss, truth, preds = evaluate(model, val_loader, criterion, device)
            val_accuracy = accuracy_score(truth, preds)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_accuracy"].append(val_accuracy)
            mlflow.log_metrics(
                {"train_loss": train_loss, "val_loss": val_loss, "val_accuracy": val_accuracy},
                step=epoch + 1,
            )
            print(
                f"Epoch {epoch + 1}/{args.epochs}: train_loss={train_loss:.4f}, "
                f"val_loss={val_loss:.4f}, val_accuracy={val_accuracy:.4f}"
            )
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

        if best_state is None:
            raise RuntimeError("Training did not produce a model state")
        model.load_state_dict(best_state)
        model.to(device)
        val_loss, truth, preds = evaluate(model, val_loader, criterion, device)
        matrix = confusion_matrix(truth, preds, labels=[0, 1])
        report = classification_report(truth, preds, target_names=class_names, output_dict=True, zero_division=0)
        elapsed_seconds = time.perf_counter() - started
        final_metrics = {
            "accuracy": float(accuracy_score(truth, preds)),
            "precision_macro": float(precision_score(truth, preds, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(truth, preds, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(truth, preds, average="macro", zero_division=0)),
            "validation_loss": float(val_loss),
            "training_seconds": elapsed_seconds,
            "run_id": run.info.run_id,
            "classification_report": report,
            "confusion_matrix": matrix.tolist(),
            "history": history,
        }
        mlflow.log_metrics({key: value for key, value in final_metrics.items() if isinstance(value, float)})

        loss_curve_path = figures_dir / "loss_curves.png"
        confusion_path = figures_dir / "confusion_matrix.png"
        save_curves(history, loss_curve_path)
        save_confusion_matrix(matrix, class_names, confusion_path)
        metrics_out.write_text(json.dumps(final_metrics, indent=2), encoding="utf-8")

        model_version = run.info.run_id[:8]
        save_model_bundle(
            args.model_out,
            model.cpu(),
            class_names=class_names,
            image_size=args.image_size,
            architecture=args.architecture,
            metadata={
                "model_version": model_version,
                "mlflow_run_id": run.info.run_id,
                "validation_accuracy": final_metrics["accuracy"],
                "validation_f1_macro": final_metrics["f1_macro"],
                "training_timestamp_unix": time.time(),
                "dataset_manifest": str(args.manifest),
            },
        )
        mlflow.log_artifact(args.model_out, artifact_path="model")
        mlflow.log_artifact(str(metrics_out), artifact_path="evaluation")
        mlflow.log_artifact(str(loss_curve_path), artifact_path="figures")
        mlflow.log_artifact(str(confusion_path), artifact_path="figures")
        mlflow.log_dict(report, "evaluation/classification_report.json")
        mlflow.set_tag("stage", "baseline")
        mlflow.set_tag("use_case", "pet-adoption-cats-vs-dogs")
        print(json.dumps(final_metrics, indent=2))


if __name__ == "__main__":
    main()
