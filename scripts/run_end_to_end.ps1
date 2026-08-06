$ErrorActionPreference = "Stop"
& .\.venv\Scripts\Activate.ps1

dvc repro
pytest
ruff check src tests
docker build -t cats-dogs-mlops:local .
docker compose -f deployment/docker-compose.yml up -d --build

$healthy = $false
for ($i = 1; $i -le 30; $i++) {
    try {
        $response = Invoke-RestMethod -Uri "http://localhost:8000/health" -TimeoutSec 5
        if ($response.status -eq "healthy") { $healthy = $true; break }
    } catch { Start-Sleep -Seconds 2 }
}
if (-not $healthy) {
    docker compose -f deployment/docker-compose.yml logs api
    throw "API did not become healthy"
}

python -m cats_dogs_mlops.smoke_test --base-url http://localhost:8000 --image tests/assets/cat_demo.png
python -m cats_dogs_mlops.post_deploy_batch --base-url http://localhost:8000 --limit 20
Write-Host "API: http://localhost:8000/docs"
Write-Host "MLflow: http://localhost:5000"
Write-Host "Prometheus: http://localhost:9090"
Write-Host "Grafana: http://localhost:3000 (admin/admin)"
