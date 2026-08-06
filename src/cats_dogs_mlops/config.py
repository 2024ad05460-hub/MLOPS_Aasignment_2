from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    model_path: Path = Path(os.getenv("MODEL_PATH", PROJECT_ROOT / "models" / "model.pt"))
    monitoring_db: Path = Path(os.getenv("MONITORING_DB", PROJECT_ROOT / "monitoring" / "predictions.db"))
    image_size: int = int(os.getenv("IMAGE_SIZE", "224"))
    class_names: tuple[str, str] = ("cat", "dog")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")


settings = Settings()
