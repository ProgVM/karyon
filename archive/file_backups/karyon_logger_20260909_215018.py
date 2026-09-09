# karyon_logger.py
import logging
import sys
import os

def get_logger():
    """Configures and returns the unified logger with line-buffered stdout and disk log streaming."""
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)
        
    os.makedirs("logs", exist_ok=True)
    log_file = os.path.join("logs", "train.log")

    logger = logging.getLogger("karyon")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter("%(module)-15s | %(levelname)-8s | %(asctime)s | %(message)s")

        stream_h = logging.StreamHandler(sys.stdout)
        stream_h.setFormatter(formatter)
        logger.addHandler(stream_h)

        file_h = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_h.setFormatter(formatter)
        logger.addHandler(file_h)

        # Also configure root logger for dependencies
        logging.basicConfig(
            level=logging.INFO,
            format="%(module)-15s | %(levelname)-8s | %(asctime)s | %(message)s",
            handlers=[stream_h, file_h]
        )
    return logger
