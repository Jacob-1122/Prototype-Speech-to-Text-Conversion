import os
import logging

logger = logging.getLogger(__name__)

def setup_directories():
    """Ensure that all necessary directories exist."""
    dirs = [
        "data",
        os.path.join("data", "atc_dataset"),
        os.path.join("data", "noise_samples"),
        os.path.join("data", "augmented"),
        "models",
        os.path.join("models", "atc_whisper")
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        logger.info(f"Directory ensured: {d}")
