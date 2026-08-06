from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-deployment health and prediction smoke tests.")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--image", default="tests/assets/cat_demo.png")
    args = parser.parse_args()

    health = requests.get(f"{args.base_url}/health", timeout=15)
    health.raise_for_status()
    health_payload = health.json()
    if health_payload.get("status") != "healthy" or not health_payload.get("model_loaded"):
        raise SystemExit(f"Health check failed: {health_payload}")

    image_path = Path(args.image)
    if not image_path.exists():
        raise SystemExit(f"Smoke-test image does not exist: {image_path}")
    with image_path.open("rb") as handle:
        prediction = requests.post(
            f"{args.base_url}/predict",
            files={"file": (image_path.name, handle, "image/png")},
            timeout=30,
        )
    prediction.raise_for_status()
    payload = prediction.json()
    required = {"request_id", "label", "confidence", "probabilities", "latency_ms"}
    if missing := required.difference(payload):
        raise SystemExit(f"Prediction response missing fields: {sorted(missing)}")
    if payload["label"] not in {"cat", "dog"}:
        raise SystemExit(f"Unexpected label: {payload['label']}")
    if abs(sum(payload["probabilities"].values()) - 1.0) > 1e-3:
        raise SystemExit(f"Probabilities do not sum to one: {payload['probabilities']}")
    print(json.dumps({"health": health_payload, "prediction": payload}, indent=2))


if __name__ == "__main__":
    main()
