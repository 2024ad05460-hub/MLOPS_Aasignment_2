from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from PIL import Image, UnidentifiedImageError
import torch

from .data import preprocess_image
from .model import ModelBundle, load_model_bundle


@dataclass(frozen=True)
class Prediction:
    label: str
    probabilities: dict[str, float]
    confidence: float


def predict_tensor(model: torch.nn.Module, tensor: torch.Tensor, class_names: list[str]) -> Prediction:
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    with torch.inference_mode():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0].detach().cpu()
    probabilities = {name: float(probs[index].item()) for index, name in enumerate(class_names)}
    best_index = int(torch.argmax(probs).item())
    return Prediction(
        label=class_names[best_index],
        probabilities=probabilities,
        confidence=float(probs[best_index].item()),
    )


def predict_image(image: Image.Image, bundle: ModelBundle) -> Prediction:
    tensor = preprocess_image(image, bundle.image_size)
    return predict_tensor(bundle.model, tensor, bundle.class_names)


def predict_bytes(data: bytes, bundle: ModelBundle) -> Prediction:
    try:
        with Image.open(BytesIO(data)) as image:
            return predict_image(image, bundle)
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("Uploaded file is not a readable image") from exc


def load_predictor(model_path: str | Path) -> ModelBundle:
    return load_model_bundle(model_path, device="cpu")
