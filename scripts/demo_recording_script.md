# Screen-recording script (target duration: 4 minutes 30 seconds)

## 0:00-0:25 - Repository and versioning
Show the repository tree, `git log --oneline -3`, `dvc.yaml`, `params.yaml`, and `dvc status`. State that Git versions code/configuration and DVC versions raw/processed data plus model outputs.

## 0:25-1:10 - Reproducible model pipeline and MLflow
Run `dvc repro` (or show a completed run), open `http://localhost:5000`, select the latest experiment, and show parameters, accuracy/F1, `loss_curves.png`, `confusion_matrix.png`, and `model.pt` artifact.

## 1:10-1:45 - Automated tests and container image
Run `pytest -q`, then `docker build -t cats-dogs-mlops:demo .`. Show the successful test count and Docker image.

## 1:45-2:35 - CI/CD workflow
Open `.github/workflows/ci-cd.yml`. Highlight checkout, dependency installation, Ruff, pytest, Docker build/push to GHCR, main-branch deployment, and post-deployment smoke test. Briefly show a successful Actions run if available.

## 2:35-3:25 - Deployment and prediction
Run `docker compose -f deployment/docker-compose.yml up -d`. Open `http://localhost:8000/docs`, call `/health`, and upload `tests/assets/cat_demo.png` to `/predict`. Show class probabilities, label, request ID, latency, and model version.

## 3:25-4:05 - Monitoring and logs
Open Grafana at `http://localhost:3000` and show request count, p95 latency, error rate, throughput, and class distribution. Also run `docker logs cats-dogs-api --tail 10` to show JSON request logs without image contents.

## 4:05-4:30 - Post-deployment performance
Run `python -m cats_dogs_mlops.post_deploy_batch --limit 20`, then open `/monitoring/performance`. Show labeled sample count and accuracy. End by showing the submission ZIP contents.
