"""
This module sets up logging for the application. 
It creates a directory for logs, generates a log file with a timestamp, 
and configures the logging settings.
"""

import logging
import os
from datetime import datetime

# Directory to store log files
LOG_DIR = os.path.join(os.getcwd(), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# Log file name with timestamp
CURRENT_TIME_STAMP = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
LOG_FILE_NAME = f"log_{CURRENT_TIME_STAMP}.log"
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE_NAME)

# Configure logging
logging.basicConfig(
    filename=LOG_FILE_PATH,
    filemode='w',
    format='[%(asctime)s] %(name)s - %(levelname)s - %(message)s',
    level=logging.DEBUG  # Use DEBUG for detailed logs
)

# Create a logger instance
logger = logging.getLogger("job_recommender_logger")
