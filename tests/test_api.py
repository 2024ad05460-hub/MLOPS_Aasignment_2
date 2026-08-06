from fastapi.testclient import TestClient
from PIL import Image

from cats_dogs_mlops.model import TinyCNN, save_model_bundle


def test_health_and_prediction_endpoints(tmp_path, monkeypatch) -> None:
    model_path = tmp_path / "model.pt"
    save_model_bundle(model_path, TinyCNN(), ["cat", "dog"], 224, "tinycnn", {"model_version": "test"})
    monkeypatch.setenv("MODEL_PATH", str(model_path))
    monkeypatch.setenv("MONITORING_DB", str(tmp_path / "monitoring.db"))

    # Import after environment variables are set so application settings use the test paths.
    import cats_dogs_mlops.api as api_module

    image_path = tmp_path / "cat.png"
    Image.new("RGB", (224, 224), (200, 120, 80)).save(image_path)

    with TestClient(api_module.app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["status"] == "healthy"
        with image_path.open("rb") as handle:
            response = client.post("/predict", files={"file": ("cat.png", handle, "image/png")})
        assert response.status_code == 200
        payload = response.json()
        assert payload["label"] in {"cat", "dog"}
        assert abs(sum(payload["probabilities"].values()) - 1.0) < 1e-3
