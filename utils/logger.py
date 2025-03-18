import logging
import os
from logging.handlers import RotatingFileHandler

# Ensure logs directory exists
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "app.log")

# Create a custom logger
logger = logging.getLogger("AppLogger")
logger.setLevel(logging.DEBUG)  # Capture all levels

# Log format
formatter = logging.Formatter("%(asctime)s - %(name)s - [%(levelname)s] - %(filename)s:%(lineno)d - %(message)s")

# File Handler (Rotating logs)
file_handler = RotatingFileHandler(LOG_FILE, maxBytes=5_000_000, backupCount=5)
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.DEBUG)

# Console Handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
console_handler.setLevel(logging.INFO)

# Add handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

def get_logger():
    """Returns the configured logger instance."""
    return logger
