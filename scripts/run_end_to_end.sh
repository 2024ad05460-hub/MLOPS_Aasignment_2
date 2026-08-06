#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate

dvc repro
pytest
ruff check src tests
docker build -t cats-dogs-mlops:local .
docker compose -f deployment/docker-compose.yml up -d --build

for attempt in $(seq 1 30); do
  if curl -fsS http://localhost:8000/health >/dev/null; then
    break
  fi
  sleep 2
  if [ "$attempt" = "30" ]; then
    docker compose -f deployment/docker-compose.yml logs api
    exit 1
  fi
done

python -m cats_dogs_mlops.smoke_test --base-url http://localhost:8000 --image tests/assets/cat_demo.png
python -m cats_dogs_mlops.post_deploy_batch --base-url http://localhost:8000 --limit 20

echo "API: http://localhost:8000/docs"
echo "MLflow: http://localhost:5000"
echo "Prometheus: http://localhost:9090"
echo "Grafana: http://localhost:3000 (admin/admin)"
