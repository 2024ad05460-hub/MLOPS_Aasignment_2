#!/usr/bin/env bash
set -euo pipefail

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

if [ ! -f data/processed/manifest.csv ]; then
  python -m cats_dogs_mlops.generate_demo_data --output data/raw/demo --per-class 40
  python -m cats_dogs_mlops.prepare_data --raw-dir data/raw --processed-dir data/processed --image-size 224 --train-ratio 0.8 --val-ratio 0.1 --seed 42
fi

python -m cats_dogs_mlops.train --manifest data/processed/manifest.csv --epochs 5 --architecture tinycnn
python -m cats_dogs_mlops.evaluate
pytest

echo "Bootstrap complete. Start services with: docker compose -f deployment/docker-compose.yml up -d --build"
