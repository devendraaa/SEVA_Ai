import logging
import time

from langchain_chroma import Chroma

logger = logging.getLogger(__name__)

def retrieve_scheme_details(
    scheme_id: str,
    detail_db: Chroma,
) -> dict:
    """Fetch all child chunks belonging to a scheme."""

    start_time = time.perf_counter()

    try:
        data = detail_db.get(
            where={"scheme_id": scheme_id},
            include=["documents", "metadatas"],
        )

        documents = [
            {
                "page_content": content,
                "metadata": metadata or {},
            }
            for content, metadata in zip(
                data["documents"],
                data["metadatas"],
            )
            if content
        ]

        latency = time.perf_counter() - start_time

        logger.info(
            "Scheme detail retrieval completed | "
            "scheme_id=%s | chunks=%d | latency=%.4fs",
            scheme_id,
            len(documents),
            latency,
        )

        return {
            "scheme_id": scheme_id,
            "retrieved_chunks": len(documents),
            "documents": documents,
            "latency": latency,
        }

    except Exception:
        logger.exception(
            "Scheme detail retrieval failed | scheme_id=%s",
            scheme_id,
        )
        raise
