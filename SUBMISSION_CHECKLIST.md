# Assignment Submission Checklist

## M1 - Model development and tracking (10/10 coverage)
- [x] Git-ready repository structure and `.gitignore`
- [x] DVC initialization/configuration, `dvc.yaml`, `params.yaml`
- [x] Kaggle CLI downloader and dataset extraction procedure
- [x] Image validation, EXIF handling, RGB conversion, 224x224 resize
- [x] Stratified 80/10/10 train/validation/test split
- [x] Data augmentation
- [x] TinyCNN baseline and optional MobileNetV3 transfer learning
- [x] PyTorch `.pt` model artifact
- [x] MLflow parameters, metrics, tags, model, JSON results, confusion matrix, loss curves

## M2 - Packaging and containerization (10/10 coverage)
- [x] FastAPI `/health` endpoint
- [x] FastAPI `/predict` endpoint returning label and class probabilities
- [x] Additional `/model-info`, `/feedback`, `/monitoring/performance`, `/metrics`
- [x] Pinned training, API, and CI requirements
- [x] Non-root Dockerfile with health check
- [x] Curl/Postman/Swagger instructions

## M3 - CI (10/10 coverage)
- [x] Data preprocessing unit test
- [x] Model/inference unit test
- [x] Monitoring utility test
- [x] API integration test
- [x] GitHub Actions checkout, setup, dependency cache, Ruff, pytest
- [x] Docker Buildx image build
- [x] GHCR image push
- [x] JUnit, deployment logs, SBOM, and provenance evidence

## M4 - CD and deployment (10/10 coverage)
- [x] Docker Compose target
- [x] Kubernetes Deployment + Service + HPA alternative
- [x] Main-branch image pull and deploy
- [x] Optional persistent VM deployment through SSH
- [x] Health smoke test
- [x] Prediction smoke test
- [x] Pipeline failure on smoke-test error

## M5 - Monitoring and final submission (10/10 coverage)
- [x] Structured request/response metadata logging, no image bytes
- [x] Prometheus request count, latency, error and prediction metrics
- [x] Provisioned Grafana dashboard
- [x] SQLite prediction metadata store
- [x] Delayed true-label feedback endpoint
- [x] Batch post-deployment performance collection
- [x] Consolidated source/config/model/report ZIP
- [x] Under-five-minute screen-recording script

## Evidence to replace after the full Kaggle run
- [ ] Kaggle DVC hash and `dvc status` screenshot
- [ ] MLflow full-dataset run ID and metrics screenshot
- [ ] Full-dataset confusion matrix and loss curve
- [ ] GitHub Actions successful CI/CD run screenshot
- [ ] Grafana screenshot after test traffic
- [ ] Less-than-five-minute screen recording file
