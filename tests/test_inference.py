import torch

from cats_dogs_mlops.inference import predict_tensor
from cats_dogs_mlops.model import TinyCNN


def test_predict_tensor_returns_valid_probability_distribution() -> None:
    torch.manual_seed(1)
    model = TinyCNN(num_classes=2).eval()
    tensor = torch.zeros(3, 224, 224)
    prediction = predict_tensor(model, tensor, ["cat", "dog"])
    assert prediction.label in {"cat", "dog"}
    assert set(prediction.probabilities) == {"cat", "dog"}
    assert abs(sum(prediction.probabilities.values()) - 1.0) < 1e-6
    assert 0.0 <= prediction.confidence <= 1.0
