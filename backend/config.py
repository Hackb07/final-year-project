"""Application configuration and logging setup.

All paths are resolved relative to the project root (BASE_DIR) so the
application is OS-agnostic and contains no hard-coded Windows paths.
Values come from environment variables / a `.env` file at the project root.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

# Project root = parent of the `backend/` directory containing this module.
BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")


class Settings:
    """Runtime settings loaded from environment variables / .env file."""

    def __init__(self) -> None:
        # --- General ---
        self.app_env: str = os.getenv("APP_ENV", "development")

        # --- Computer vision ---
        # Relative paths are resolved against BASE_DIR.
        self.yolo_model_path: Path = BASE_DIR / os.getenv(
            "YOLO_MODEL_PATH", "models/capsule_yolo11.pt"
        )
        try:
            self.confidence_threshold: float = float(
                os.getenv("CONFIDENCE_THRESHOLD", "0.40")
            )
        except ValueError:
            self.confidence_threshold = 0.40
        device_env = os.getenv("DEVICE", "").strip().lower()
        self.device: Optional[str] = device_env or None

        # --- Uploads / storage ---
        try:
            self.max_upload_size_mb: int = int(
                os.getenv("MAX_UPLOAD_SIZE_MB", "500")
            )
        except ValueError:
            self.max_upload_size_mb = 500
        self.max_upload_bytes: int = self.max_upload_size_mb * 1024 * 1024

        self.upload_dir: Path = BASE_DIR / os.getenv("UPLOAD_DIR", "data/uploads")
        self.output_dir: Path = BASE_DIR / os.getenv("OUTPUT_DIR", "outputs")
        self.original_dir: Path = self.output_dir / "original"
        self.annotated_dir: Path = self.output_dir / "annotated"
        self.log_dir: Path = BASE_DIR / os.getenv("LOG_DIR", "logs")

        # --- LLM / agents (Groq, OpenAI-compatible) ---
        self.groq_api_key: Optional[str] = os.getenv("GROQ_API_KEY") or None
        self.groq_model: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
        self.groq_base_url: str = os.getenv(
            "GROQ_BASE_URL", "https://api.groq.com/openai/v1"
        )

        # --- API ---
        raw_origins = os.getenv("CORS_ORIGINS", "*")
        self.cors_origins: List[str] = [o.strip() for o in raw_origins.split(",") if o.strip()]

        # --- Allowed media types ---
        self.allowed_image_extensions: tuple = (
            ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff",
        )

        self.ensure_directories()

    def ensure_directories(self) -> None:
        """Create the storage directories the app relies on."""
        for d in (
            self.upload_dir,
            self.original_dir,
            self.annotated_dir,
            self.log_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()


def setup_logging(log_dir: Optional[Path] = None) -> None:
    """Configure rotating file loggers (application.log, cv.log) + console."""
    log_dir = log_dir or settings.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    def _configure(name: str, filename: str, level: int) -> logging.Logger:
        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.propagate = False

        if not any(isinstance(h, logging.handlers.RotatingFileHandler) and getattr(h, "baseFilename", "").endswith(filename) for h in logger.handlers):
            handler = logging.handlers.RotatingFileHandler(
                log_dir / filename, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
            )
            handler.setFormatter(fmt)
            logger.addHandler(handler)
        return logger

    _configure("application", "application.log", logging.INFO)
    _configure("cv", "cv.log", logging.DEBUG)

    # Root console handler (also catches anything unhandled).
    root = logging.getLogger()
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        console = logging.StreamHandler()
        console.setFormatter(fmt)
        root.addHandler(console)
    root.setLevel(logging.INFO)

    # Surface 'application' + 'cv' logs to root so they also appear on console.
    logging.getLogger("cv").propagate = True


setup_logging()
