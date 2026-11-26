
import logging
import os
from functools import lru_cache

from langfuse import get_client

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_langfuse_client():
    """Return the shared Langfuse client."""

    try:
        required_keys = [
            "LANGFUSE_PUBLIC_KEY",
            "LANGFUSE_SECRET_KEY",
        ]

        missing_keys = [
            key for key in required_keys if not os.getenv(key)
        ]

        if missing_keys:
            raise EnvironmentError(
                f"Missing Langfuse environment variables: {missing_keys}"
            )

        client = get_client()

        logger.info("Langfuse client initialized successfully.")

        return client

    except Exception:
        logger.exception("Failed to initialize Langfuse client.")
        raise
