
import logging
from pathlib import Path

from dotenv import load_dotenv

from core.logging import setup_logging
from observability.langfuse_client import get_langfuse_client

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def initialize_seva():
    """Initialize environment, logging and Langfuse."""

    try:
        # 1. Load environment variables
        load_dotenv(PROJECT_ROOT / ".env")

        # 2. Initialize application logging
        logger = setup_logging()

        # 3. Initialize Langfuse
        langfuse = get_langfuse_client()

        logger.info("SEVA_Ai startup completed successfully.")

        return langfuse

    except Exception:
        logging.exception("SEVA_Ai initialization failed.")
        raise
