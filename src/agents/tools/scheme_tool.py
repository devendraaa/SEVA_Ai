import logging

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


@tool
def search_schemes(query: str) -> str:
    """Search government schemes using a user query."""

    try:
        # Temporary mock implementation.
        # We will connect existing hybrid retrieval in Milestone 4.

        logger.info("Scheme search tool called | query=%s", query)

        return f"Mock scheme search completed for: {query}"

    except Exception:
        logger.exception("Scheme search tool failed.")
        raise