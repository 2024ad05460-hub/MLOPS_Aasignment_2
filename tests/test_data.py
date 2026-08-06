from PIL import Image
import torch

from cats_dogs_mlops.data import preprocess_image


def test_preprocess_image_returns_normalized_rgb_tensor() -> None:
    image = Image.new("L", (80, 120), color=128)
    tensor = preprocess_image(image, image_size=224)
    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32
    assert torch.isfinite(tensor).all()
    assert float(tensor.abs().max()) < 5.0
