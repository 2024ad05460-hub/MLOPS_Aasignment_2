from __future__ import annotations

import argparse
import json
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .data import ManifestDataset
from .model import build_model, save_model_bundle


def str_to_bool(value: str | bool) -> bool:
    """Parse command-line Boolean values safely."""
    if isinstance(value, bool):
        return value

    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False

    raise argparse.ArgumentTypeError(
        f"Expected a Boolean value, received: {value!r}"
    )


def set_seed(seed: int) -> None:
    """Set reproducible random seeds."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Benchmarking improves fixed-size CNN throughput.
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = True


def choose_device(requested_device: str) -> torch.device:
    """Choose CUDA automatically unless CPU is explicitly requested."""
    if requested_device == "cpu":
        return torch.device("cpu")

    if requested_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "--device cuda was requested, but CUDA is unavailable."
            )
        return torch.device("cuda")

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_grad_scaler(enabled: bool) -> Any:
    """Create a GradScaler compatible with recent and older PyTorch versions."""
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast_context(device: torch.device, enabled: bool):
    """Return the appropriate automatic mixed-precision context."""
    if not enabled:
        return nullcontext()

    try:
        return torch.amp.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=True,
        )
    except AttributeError:
        return torch.cuda.amp.autocast(enabled=True)


def create_loader(
    dataset: ManifestDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    """Build a Windows-safe DataLoader."""
    persistent_workers = num_workers > 0

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=persistent_workers,
        drop_last=False,
    )


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp_enabled: bool,
) -> tuple[float, list[int], list[int], list[float]]:
    """Evaluate the model and return loss, labels, predictions and dog probabilities."""
    model.eval()

    total_loss = 0.0
    total_samples = 0
    truth: list[int] = []
    predictions: list[int] = []
    positive_probabilities: list[float] = []

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast_context(device, amp_enabled):
                logits = model(images)
                loss = criterion(logits, labels)

            probabilities = torch.softmax(logits.float(), dim=1)
            batch_size = labels.size(0)

            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size

            predictions.extend(torch.argmax(probabilities, dim=1).cpu().tolist())
            positive_probabilities.extend(probabilities[:, 1].cpu().tolist())
            truth.extend(labels.cpu().tolist())

    mean_loss = total_loss / max(total_samples, 1)
    return mean_loss, truth, predictions, positive_probabilities


def save_training_curves(
    history: dict[str, list[float]],
    output_path: Path,
) -> None:
    """Save loss and validation-accuracy curves in one figure."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)

    figure, first_axis = plt.subplots(figsize=(9, 5))
    first_axis.plot(epochs, history["train_loss"], label="Train loss")
    first_axis.plot(epochs, history["val_loss"], label="Validation loss")
    first_axis.set_xlabel("Epoch")
    first_axis.set_ylabel("Cross-entropy loss")
    first_axis.legend(loc="upper left")

    second_axis = first_axis.twinx()
    second_axis.plot(
        epochs,
        history["val_accuracy"],
        linestyle="--",
        label="Validation accuracy",
    )
    second_axis.set_ylabel("Validation accuracy")
    second_axis.set_ylim(0.0, 1.0)
    second_axis.legend(loc="upper right")

    plt.title("Training history")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_confusion_matrix(
    matrix: np.ndarray,
    class_names: list[str],
    output_path: Path,
) -> None:
    """Save an annotated validation confusion matrix."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(5, 4))
    image = axis.imshow(matrix)
    figure.colorbar(image, ax=axis)

    axis.set_title("Validation confusion matrix")
    axis.set_xticks(range(len(class_names)), class_names)
    axis.set_yticks(range(len(class_names)), class_names)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")

    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
            )

    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_roc_curve(
    truth: list[int],
    probabilities: list[float],
    output_path: Path,
) -> float:
    """Save ROC curve and return ROC-AUC."""
    false_positive_rate, true_positive_rate, _ = roc_curve(
        truth,
        probabilities,
    )
    roc_auc = float(auc(false_positive_rate, true_positive_rate))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6, 5))
    axis.plot(
        false_positive_rate,
        true_positive_rate,
        label=f"ROC-AUC = {roc_auc:.4f}",
    )
    axis.plot([0, 1], [0, 1], linestyle="--")
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")
    axis.set_title("Validation ROC curve")
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)

    return roc_auc


def save_precision_recall_curve(
    truth: list[int],
    probabilities: list[float],
    output_path: Path,
) -> float:
    """Save precision-recall curve and return average precision."""
    precision_values, recall_values, _ = precision_recall_curve(
        truth,
        probabilities,
    )
    average_precision = float(
        average_precision_score(truth, probabilities)
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6, 5))
    axis.plot(
        recall_values,
        precision_values,
        label=f"Average precision = {average_precision:.4f}",
    )
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_title("Validation precision-recall curve")
    axis.legend(loc="lower left")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)

    return average_precision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a baseline or transfer-learning cats-vs-dogs model."
    )

    parser.add_argument(
        "--manifest",
        default="data/processed/manifest.csv",
    )
    parser.add_argument(
        "--model-out",
        default="models/model.pt",
    )
    parser.add_argument(
        "--metrics-out",
        default="reports/train_metrics.json",
    )
    parser.add_argument(
        "--figures-dir",
        default="figures",
    )
    parser.add_argument(
        "--architecture",
        choices=["tinycnn", "mobilenet_v3_small"],
        default="mobilenet_v3_small",
    )
    parser.add_argument(
        "--pretrained",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=True,
    )
    parser.add_argument(
        "--freeze-backbone",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=False,
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    parser.add_argument(
        "--amp",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=True,
        help="Enable CUDA automatic mixed precision.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=4,
        help="Early-stopping patience.",
    )
    parser.add_argument(
        "--scheduler-patience",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--scheduler-factor",
        type=float,
        default=0.3,
    )
    parser.add_argument(
        "--min-learning-rate",
        type=float,
        default=1e-7,
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--gradient-clip",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--experiment-name",
        default="cats-dogs-classification",
    )
    parser.add_argument(
        "--tracking-uri",
        default=os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = choose_device(args.device)
    amp_enabled = bool(args.amp and device.type == "cuda")
    class_names = ["cat", "dog"]

    print("=" * 72)
    print("Cats vs Dogs training")
    print("=" * 72)
    print(f"Device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(
        "GPU: "
        + (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "Not available"
        )
    )
    print(f"AMP enabled: {amp_enabled}")
    print(f"Architecture: {args.architecture}")
    print(f"Pretrained: {args.pretrained}")
    print(f"Freeze backbone: {args.freeze_backbone}")

    train_dataset = ManifestDataset(
        args.manifest,
        split="train",
        image_size=args.image_size,
        augment=True,
    )
    validation_dataset = ManifestDataset(
        args.manifest,
        split="val",
        image_size=args.image_size,
        augment=False,
    )

    train_loader = create_loader(
        train_dataset,
        args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        device=device,
    )
    validation_loader = create_loader(
        validation_dataset,
        args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        device=device,
    )

    model = build_model(
        args.architecture,
        num_classes=2,
        pretrained=args.pretrained,
    )

    if (
        args.architecture == "mobilenet_v3_small"
        and args.freeze_backbone
    ):
        for parameter in model.features.parameters():
            parameter.requires_grad = False

    model = model.to(device)

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    criterion = nn.CrossEntropyLoss(
        label_smoothing=args.label_smoothing
    )

    optimizer = torch.optim.AdamW(
        (
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
        min_lr=args.min_learning_rate,
    )

    scaler = make_grad_scaler(amp_enabled)

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    figures_directory = Path(args.figures_dir)
    metrics_output = Path(args.metrics_out)
    model_output = Path(args.model_out)

    figures_directory.mkdir(parents=True, exist_ok=True)
    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    model_output.parent.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run() as run:
        mlflow.log_params(
            {
                "architecture": args.architecture,
                "pretrained": args.pretrained,
                "freeze_backbone": args.freeze_backbone,
                "epochs_requested": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "image_size": args.image_size,
                "seed": args.seed,
                "device": str(device),
                "gpu_name": (
                    torch.cuda.get_device_name(0)
                    if torch.cuda.is_available()
                    else "none"
                ),
                "amp_enabled": amp_enabled,
                "label_smoothing": args.label_smoothing,
                "early_stopping_patience": args.patience,
                "scheduler_patience": args.scheduler_patience,
                "scheduler_factor": args.scheduler_factor,
                "gradient_clip": args.gradient_clip,
                "train_samples": len(train_dataset),
                "validation_samples": len(validation_dataset),
                "total_parameters": total_parameters,
                "trainable_parameters": trainable_parameters,
                "torch_version": torch.__version__,
            }
        )

        history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "learning_rate": [],
        }

        best_accuracy = -1.0
        best_epoch = 0
        best_state: dict[str, torch.Tensor] | None = None
        epochs_without_improvement = 0
        started = time.perf_counter()

        for epoch in range(1, args.epochs + 1):
            model.train()
            running_loss = 0.0
            processed_samples = 0

            progress = tqdm(
                train_loader,
                desc=f"Epoch {epoch}/{args.epochs}",
                leave=False,
            )

            for images, labels in progress:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with autocast_context(device, amp_enabled):
                    logits = model(images)
                    loss = criterion(logits, labels)

                scaler.scale(loss).backward()

                if args.gradient_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=args.gradient_clip,
                    )

                scaler.step(optimizer)
                scaler.update()

                batch_size = labels.size(0)
                running_loss += float(loss.item()) * batch_size
                processed_samples += batch_size

                progress.set_postfix(
                    loss=f"{running_loss / max(processed_samples, 1):.4f}"
                )

            train_loss = running_loss / max(processed_samples, 1)

            (
                validation_loss,
                truth,
                predictions,
                _,
            ) = evaluate(
                model,
                validation_loader,
                criterion,
                device,
                amp_enabled,
            )

            validation_accuracy = float(
                accuracy_score(truth, predictions)
            )

            scheduler.step(validation_accuracy)
            current_learning_rate = float(
                optimizer.param_groups[0]["lr"]
            )

            history["train_loss"].append(train_loss)
            history["val_loss"].append(validation_loss)
            history["val_accuracy"].append(validation_accuracy)
            history["learning_rate"].append(current_learning_rate)

            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "val_loss": validation_loss,
                    "val_accuracy": validation_accuracy,
                    "learning_rate": current_learning_rate,
                },
                step=epoch,
            )

            print(
                f"Epoch {epoch:02d}/{args.epochs}: "
                f"train_loss={train_loss:.4f}, "
                f"val_loss={validation_loss:.4f}, "
                f"val_accuracy={validation_accuracy:.4f}, "
                f"lr={current_learning_rate:.2e}"
            )

            if validation_accuracy > best_accuracy:
                best_accuracy = validation_accuracy
                best_epoch = epoch
                epochs_without_improvement = 0
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= args.patience:
                print(
                    "Early stopping triggered after "
                    f"{epoch} epochs. Best epoch: {best_epoch}."
                )
                break

        if best_state is None:
            raise RuntimeError("Training did not produce a model state.")

        model.load_state_dict(best_state)
        model = model.to(device)

        (
            final_validation_loss,
            truth,
            predictions,
            positive_probabilities,
        ) = evaluate(
            model,
            validation_loader,
            criterion,
            device,
            amp_enabled,
        )

        matrix = confusion_matrix(
            truth,
            predictions,
            labels=[0, 1],
        )

        report = classification_report(
            truth,
            predictions,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        )

        loss_curve_path = figures_directory / "loss_curves.png"
        confusion_path = figures_directory / "confusion_matrix.png"
        roc_path = figures_directory / "roc_curve.png"
        precision_recall_path = (
            figures_directory / "precision_recall_curve.png"
        )

        save_training_curves(history, loss_curve_path)
        save_confusion_matrix(
            matrix,
            class_names,
            confusion_path,
        )
        roc_auc = save_roc_curve(
            truth,
            positive_probabilities,
            roc_path,
        )
        average_precision = save_precision_recall_curve(
            truth,
            positive_probabilities,
            precision_recall_path,
        )

        elapsed_seconds = time.perf_counter() - started

        final_metrics: dict[str, Any] = {
            "accuracy": float(
                accuracy_score(truth, predictions)
            ),
            "precision_macro": float(
                precision_score(
                    truth,
                    predictions,
                    average="macro",
                    zero_division=0,
                )
            ),
            "recall_macro": float(
                recall_score(
                    truth,
                    predictions,
                    average="macro",
                    zero_division=0,
                )
            ),
            "f1_macro": float(
                f1_score(
                    truth,
                    predictions,
                    average="macro",
                    zero_division=0,
                )
            ),
            "roc_auc": roc_auc,
            "average_precision": average_precision,
            "validation_loss": float(final_validation_loss),
            "best_epoch": best_epoch,
            "epochs_completed": len(history["train_loss"]),
            "training_seconds": float(elapsed_seconds),
            "device": str(device),
            "gpu_name": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None
            ),
            "run_id": run.info.run_id,
            "classification_report": report,
            "confusion_matrix": matrix.tolist(),
            "history": history,
        }

        mlflow.log_metrics(
            {
                "best_val_accuracy": final_metrics["accuracy"],
                "final_val_loss": final_metrics["validation_loss"],
                "precision_macro": final_metrics["precision_macro"],
                "recall_macro": final_metrics["recall_macro"],
                "f1_macro": final_metrics["f1_macro"],
                "roc_auc": final_metrics["roc_auc"],
                "average_precision": final_metrics[
                    "average_precision"
                ],
                "training_seconds": final_metrics["training_seconds"],
                "best_epoch": float(best_epoch),
            }
        )

        metrics_output.write_text(
            json.dumps(final_metrics, indent=2),
            encoding="utf-8",
        )

        model_version = run.info.run_id[:8]

        save_model_bundle(
            model_output,
            model.cpu(),
            class_names=class_names,
            image_size=args.image_size,
            architecture=args.architecture,
            metadata={
                "model_version": model_version,
                "mlflow_run_id": run.info.run_id,
                "validation_accuracy": final_metrics["accuracy"],
                "validation_f1_macro": final_metrics["f1_macro"],
                "roc_auc": final_metrics["roc_auc"],
                "best_epoch": best_epoch,
                "training_timestamp_unix": time.time(),
                "dataset_manifest": str(args.manifest),
                "pretrained": args.pretrained,
                "freeze_backbone": args.freeze_backbone,
            },
        )

        mlflow.log_artifact(
            str(model_output),
            artifact_path="model",
        )
        mlflow.log_artifact(
            str(metrics_output),
            artifact_path="evaluation",
        )
        mlflow.log_artifact(
            str(loss_curve_path),
            artifact_path="figures",
        )
        mlflow.log_artifact(
            str(confusion_path),
            artifact_path="figures",
        )
        mlflow.log_artifact(
            str(roc_path),
            artifact_path="figures",
        )
        mlflow.log_artifact(
            str(precision_recall_path),
            artifact_path="figures",
        )
        mlflow.log_dict(
            report,
            "evaluation/classification_report.json",
        )

        mlflow.set_tags(
            {
                "stage": (
                    "transfer-learning"
                    if args.pretrained
                    else "baseline"
                ),
                "use_case": "pet-adoption-cats-vs-dogs",
                "framework": "pytorch",
                "best_epoch": str(best_epoch),
            }
        )

        print()
        print("=" * 72)
        print("Training completed successfully")
        print("=" * 72)
        print(json.dumps(final_metrics, indent=2))


if __name__ == "__main__":
    main()