"""
logger_setup.py

Configures the root logger for the project.
"""

import logging
import sys

def setup_logging(log_file_path: str):
    """Configures logging to output to both console and a file."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] - %(message)s",
        handlers=[
            logging.FileHandler(log_file_path, mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )