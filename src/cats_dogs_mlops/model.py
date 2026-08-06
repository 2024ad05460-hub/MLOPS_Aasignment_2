from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn


class TinyCNN(nn.Module):
    """Small baseline CNN that is fast enough for CI and CPU demonstrations."""

    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def build_model(architecture: str = "tinycnn", num_classes: int = 2, pretrained: bool = False) -> nn.Module:
    architecture = architecture.lower()
    if architecture == "tinycnn":
        return TinyCNN(num_classes=num_classes)

    if architecture == "mobilenet_v3_small":
        from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = mobilenet_v3_small(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model

    raise ValueError(f"Unsupported architecture: {architecture}")


@dataclass
class ModelBundle:
    model: nn.Module
    class_names: list[str]
    image_size: int
    architecture: str
    metadata: dict[str, Any]


def save_model_bundle(
    path: str | Path,
    model: nn.Module,
    class_names: list[str],
    image_size: int,
    architecture: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "class_names": class_names,
        "image_size": int(image_size),
        "architecture": architecture,
        "metadata": metadata or {},
    }
    torch.save(payload, path)


def load_model_bundle(path: str | Path, device: str | torch.device = "cpu") -> ModelBundle:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found: {path}")

    payload = torch.load(path, map_location=device, weights_only=False)
    required = {"state_dict", "class_names", "image_size", "architecture"}
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"Invalid model artifact; missing keys: {sorted(missing)}")

    model = build_model(payload["architecture"], num_classes=len(payload["class_names"]), pretrained=False)
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return ModelBundle(
        model=model,
        class_names=list(payload["class_names"]),
        image_size=int(payload["image_size"]),
        architecture=str(payload["architecture"]),
        metadata=dict(payload.get("metadata", {})),
    )
