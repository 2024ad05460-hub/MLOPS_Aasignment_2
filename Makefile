.PHONY: install demo-data prepare train evaluate test lint mlflow api docker-build compose-up compose-down smoke post-deploy dvc-run clean

install:
	python -m pip install --upgrade pip
	pip install -r requirements.txt
	pip install -e .

demo-data:
	python -m cats_dogs_mlops.generate_demo_data --output data/raw/demo --per-class 40

prepare:
	python -m cats_dogs_mlops.prepare_data --raw-dir data/raw --processed-dir data/processed --image-size 224 --train-ratio 0.8 --val-ratio 0.1 --seed 42

train:
	python -m cats_dogs_mlops.train --manifest data/processed/manifest.csv --epochs 5 --architecture tinycnn

evaluate:
	python -m cats_dogs_mlops.evaluate

test:
	pytest

lint:
	ruff check src tests

mlflow:
	mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts

api:
	uvicorn cats_dogs_mlops.api:app --reload --host 0.0.0.0 --port 8000

docker-build:
	docker build -t cats-dogs-mlops:local .

compose-up:
	docker compose -f deployment/docker-compose.yml up -d --build

compose-down:
	docker compose -f deployment/docker-compose.yml down -v

smoke:
	python -m cats_dogs_mlops.smoke_test --base-url http://localhost:8000 --image tests/assets/cat_demo.png

post-deploy:
	python -m cats_dogs_mlops.post_deploy_batch --base-url http://localhost:8000 --limit 20

dvc-run:
	dvc repro

clean:
	rm -rf data/processed/* mlruns mlartifacts monitoring/*.db .pytest_cache
