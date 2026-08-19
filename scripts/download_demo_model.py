"""Download a demo YOLO model for CapsuleAI.

IMPORTANT: This fetches the Ultralytics YOLO11-nano model pretrained on COCO
(80 everyday classes). It is a *placeholder* for development and does NOT
detect capsule endoscopy disease classes. It exists so the full pipeline
(load model -> upload -> detect -> annotate) can be exercised and verified.

To use CapsuleAI for its intended purpose, replace ``models/capsule_yolo11.pt``
with a real model trained on capsule endoscopy data (see README, section
"YOLO Model Setup").

Usage::

    python scripts/download_demo_model.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
DEST = MODELS_DIR / "capsule_yolo11.pt"
SOURCE_NAME = "yolo11n.pt"


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    source = ROOT / SOURCE_NAME
    if not source.exists():
        print(f"Downloading {SOURCE_NAME} via Ultralytics (first run only)...")
        from ultralytics import YOLO

        model = YOLO(SOURCE_NAME)  # auto-downloads the weights
        del model

    if not source.exists():
        raise SystemExit(
            "Download failed. Check your internet connection and that the "
            "Ultralytics package is installed."
        )

    shutil.copy2(source, DEST)
    print(f"Demo model ready at: {DEST}")
    print("NOTE: this is a COCO-pretrained placeholder, NOT a capsule-trained model.")


if __name__ == "__main__":
    main()
