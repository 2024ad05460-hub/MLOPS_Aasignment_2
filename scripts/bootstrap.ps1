$ErrorActionPreference = "Stop"

py -3.11 -m venv .venv
& .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

if (-not (Test-Path "data\processed\manifest.csv")) {
    python -m cats_dogs_mlops.generate_demo_data --output data/raw/demo --per-class 40
    python -m cats_dogs_mlops.prepare_data --raw-dir data/raw --processed-dir data/processed --image-size 224 --train-ratio 0.8 --val-ratio 0.1 --seed 42
}

python -m cats_dogs_mlops.train --manifest data/processed/manifest.csv --epochs 5 --architecture tinycnn
python -m cats_dogs_mlops.evaluate
pytest
Write-Host "Bootstrap complete. Start services with: docker compose -f deployment/docker-compose.yml up -d --build"
