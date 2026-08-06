$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force evidence | Out-Null

git log --oneline --decorate -10 | Out-File evidence\git_log_runtime.txt
dvc status | Out-File evidence\dvc_status_runtime.txt
pytest -q | Tee-Object evidence\pytest_runtime.txt
ruff check src tests | Tee-Object evidence\ruff_runtime.txt
Invoke-RestMethod http://localhost:8000/health | ConvertTo-Json -Depth 5 | Out-File evidence\health_runtime.json
curl.exe -s -X POST http://localhost:8000/predict -F "file=@tests/assets/cat_demo.png" | Out-File evidence\prediction_runtime.json
Invoke-RestMethod http://localhost:8000/monitoring/performance | ConvertTo-Json -Depth 5 | Out-File evidence\performance_runtime.json
docker compose -f deployment/docker-compose.yml ps | Out-File evidence\docker_compose_ps.txt
Write-Host "Evidence text files saved under evidence/. Capture UI screenshots separately."
