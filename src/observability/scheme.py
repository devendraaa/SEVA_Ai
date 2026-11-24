
import logging
from dataclasses import asdict, dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RetrievedDocument:
    """Standard representation of one retrieved item."""

    id: str
    content: str
    score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """Standard output contract for every retrieval method."""

    query: str
    method: str
    documents: list[RetrievedDocument]
    latency: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        try:
            if not self.query.strip():
                raise ValueError("Query cannot be empty.")

            if not self.method.strip():
                raise ValueError("Retrieval method cannot be empty.")

            if self.latency < 0:
                raise ValueError("Latency cannot be negative.")

            if not all(
                isinstance(doc, RetrievedDocument)
                for doc in self.documents
            ):
                raise TypeError(
                    "documents must contain RetrievedDocument objects."
                )

        except Exception:
            logger.exception("RetrievalResult validation failed.")
            raise

    def to_trace_payload(self) -> dict[str, Any]:
        """Convert retrieval output into a trace-friendly dictionary."""

        try:
            return asdict(self)

        except Exception:
            logger.exception(
                "Failed to serialize retrieval result | method=%s",
                self.method,
            )
            raise
