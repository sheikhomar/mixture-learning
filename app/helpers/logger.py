import sys, logging
from typing import Optional
from pathlib import Path
from datetime import datetime


def get_logger(name: str, file_log_dir: Optional[Path] = None) -> logging.Logger:
    """Returns a logger."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    log_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)-5.5s|%(name)-12.12s] %(message)s"
    )

    if file_log_dir is not None:
        file_log_dir.mkdir(parents=True, exist_ok=True)
        today = datetime.now().strftime("%Y-%m-%d")
        log_file = logging.FileHandler(str(file_log_dir / f"log-{today}.log"))
        log_file.setFormatter(log_formatter)
        logger.addHandler(log_file)

    log_console = logging.StreamHandler(sys.stdout)
    log_console.setFormatter(log_formatter)
    logger.addHandler(log_console)

    return logger