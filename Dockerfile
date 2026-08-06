FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_PATH=/app/models/model.pt \
    MONITORING_DB=/app/monitoring/predictions.db

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system appgroup \
    && useradd --system --gid appgroup --create-home appuser

WORKDIR /app
COPY requirements-api.txt .
RUN python -m pip install --upgrade pip \
    && pip install -r requirements-api.txt

COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-deps .
COPY models ./models
RUN mkdir -p /app/monitoring && chown -R appuser:appgroup /app

USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl --fail http://localhost:8000/health || exit 1

CMD ["uvicorn", "cats_dogs_mlops.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
