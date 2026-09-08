# karyon_logger.py
import logging
import sys

def get_logger():
    """Configures and returns the unified logger with line-buffered stdout streaming."""
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)
        
    logging.basicConfig(
        level=logging.INFO,
        format="%(module)-15s | %(levelname)-8s | %(asctime)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger("karyon")
