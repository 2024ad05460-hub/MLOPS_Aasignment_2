# End-to-End Runbook

## Fastest verification with the bundled bootstrap artifact

### Windows PowerShell

```powershell
cd Cats_Dogs_MLOps_Assignment2
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
py -3.11 -m venv .venv
& .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
pytest -q
$env:MODEL_PATH="models/model.pt"
uvicorn cats_dogs_mlops.api:app --host 0.0.0.0 --port 8000
```

In a second terminal:

```powershell
& .\.venv\Scripts\Activate.ps1
Invoke-RestMethod http://localhost:8000/health
curl.exe -X POST http://localhost:8000/predict -F "file=@tests/assets/cat_demo.png"
python -m cats_dogs_mlops.smoke_test --base-url http://localhost:8000 --image tests/assets/cat_demo.png
```

## Complete Kaggle + DVC + MLflow + Docker workflow

1. Put `kaggle.json` in `$env:USERPROFILE\.kaggle\kaggle.json` and accept the Kaggle competition rules.
2. Activate the virtual environment.
3. Download and extract the data.
4. Version the raw data with DVC.
5. Start MLflow.
6. Run `dvc repro`.
7. Test, build, deploy, smoke-test, and collect labeled production requests.

```powershell
Remove-Item -Recurse -Force data\raw\* -ErrorAction SilentlyContinue
python -m cats_dogs_mlops.download_data --mode competition --slug dogs-vs-cats --destination data/raw --force
Expand-Archive data\raw\train.zip -DestinationPath data\raw\train -Force
git init
dvc init
dvc add data/raw
git add .
git commit -m "feat: version cats-dogs MLOps pipeline and data metadata"

# Terminal 1
mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts

# Terminal 2
$env:MLFLOW_TRACKING_URI="http://localhost:5000"
dvc repro
pytest -q
ruff check src tests
docker compose -f deployment/docker-compose.yml up -d --build
python -m cats_dogs_mlops.smoke_test --base-url http://localhost:8000 --image tests/assets/cat_demo.png
python -m cats_dogs_mlops.post_deploy_batch --base-url http://localhost:8000 --manifest data/processed/manifest.csv --limit 20
```

Open:

- FastAPI Swagger: `http://localhost:8000/docs`
- MLflow: `http://localhost:5000`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000` (`admin` / `admin` for local demonstration only)

## GitHub CI/CD

Push the repository to GitHub. The included workflow runs tests, builds the image, pushes SHA/latest tags to GHCR, deploys the published image on `main`, and fails if health or prediction smoke tests fail. For a persistent VM deployment, set repository variable `ENABLE_REMOTE_DEPLOY=true` and configure `DEPLOY_HOST`, `DEPLOY_USER`, `DEPLOY_SSH_KEY`, `DEPLOY_PATH`, and `GHCR_TOKEN` secrets.

## Submission evidence

Capture MLflow run/artifacts, `dvc status`, pytest, GitHub Actions, API prediction, Grafana panels, and `/monitoring/performance`. Follow `scripts/demo_recording_script.md` for the required video under five minutes.
