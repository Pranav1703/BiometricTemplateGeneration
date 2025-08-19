import logging
import os
from datetime import datetime
from src.config import TRAINING_LOG_DIR

os.makedirs(TRAINING_LOG_DIR, exist_ok=True)

def get_logger(name: str = "train") -> logging.Logger:
    """
    Creates and configures a logger that logs both to console and a file.
    Console -> INFO+
    File    -> DEBUG+
    """
    logger = logging.getLogger(name)

    if not logger.handlers:   # Only set up once
        logger.setLevel(logging.DEBUG)

        # Timestamped run folder
        run_dir = os.path.join(TRAINING_LOG_DIR, datetime.now().strftime('%Y%m%d_%H%M%S'))
        os.makedirs(run_dir, exist_ok=True)
        log_file = os.path.join(run_dir, f"{name}.log")

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))

        # File handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s"))

        logger.addHandler(ch)
        logger.addHandler(fh)
        logger.propagate = False  # Avoid duplicate logs from root logger

    return logger
