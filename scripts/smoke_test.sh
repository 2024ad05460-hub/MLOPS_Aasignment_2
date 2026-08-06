#!/usr/bin/env bash
set -euo pipefail
BASE_URL="${1:-http://localhost:8000}"
IMAGE="${2:-tests/assets/cat_demo.png}"
curl -fsS "$BASE_URL/health"
curl -fsS -X POST "$BASE_URL/predict" -F "file=@$IMAGE"
python -m cats_dogs_mlops.smoke_test --base-url "$BASE_URL" --image "$IMAGE"
