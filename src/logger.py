import logging
import sys
from pathlib import Path


def get_logger(
    name: str, level=logging.INFO, log_to_file: bool = False, log_dir: str = "logs"
) -> logging.Logger:
    """
    Configures and returns a logger with console (and optional file) output.

    Args:
        name (str): Logger name (typically __name__).
        level (int): Logging level (e.g., logging.INFO).
        log_to_file (bool): If True, logs will also be written to a file.
        log_dir (str): Directory for log files if log_to_file is True.

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.hasHandlers():
        return logger  # Prevent duplicate handlers on reload

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Optional file handler
    if log_to_file:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(f"{log_dir}/{name}.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
