
import logging
from contextlib import contextmanager
from typing import Any, Generator

from observability.langfuse_client import get_langfuse_client

logger = logging.getLogger(__name__)


@contextmanager
def trace_span(
    name: str,
    input_data: Any = None,
    metadata: dict | None = None,
) -> Generator[Any, None, None]:
    """Create a Langfuse observation without breaking application execution."""

    try:
        langfuse = get_langfuse_client()

        observation = langfuse.start_as_current_observation(
            as_type="span",
            name=name,
            input=input_data,
            metadata=metadata,
        )

    except Exception:
        logger.exception("Could not start Langfuse observation: %s", name)
        yield None
        return

    # Application exceptions propagate normally.
    # Langfuse records them on the observation.
    with observation as span:
        yield span
