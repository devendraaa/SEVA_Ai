
import logging
import time

from langchain_chroma import Chroma

logger = logging.getLogger(__name__)

RETRIEVAL_K = 30


def retrieve_dense_schemes(
    query: str,
    discovery_db: Chroma,
    k: int = RETRIEVAL_K,
) -> dict:
    """Retrieve schemes using dense vector search."""

    start_time = time.perf_counter()

    try:
        results = discovery_db.similarity_search_with_score(
            query=query,
            k=k,
        )

        ranked_schemes = []
        seen_scheme_ids = set()

        for chunk_rank, (doc, score) in enumerate(results, start=1):
            scheme_id = doc.metadata.get("scheme_id")

            if scheme_id and scheme_id not in seen_scheme_ids:
                seen_scheme_ids.add(scheme_id)

                ranked_schemes.append({
                    "scheme_id": scheme_id,
                    "rank": len(ranked_schemes) + 1,
                    "chunk_rank": chunk_rank,
                    "distance": float(score),
                    "section": doc.metadata.get("child_type"),
                })

        latency = time.perf_counter() - start_time

        logger.info(
            "Dense retrieval completed | chunks=%d | schemes=%d | latency=%.4fs",
            len(results),
            len(ranked_schemes),
            latency,
        )

        return {
            "query": query,
            "retrieved_chunks": len(results),
            "retrieved_schemes": ranked_schemes,
            "latency": latency,
        }

    except Exception:
        logger.exception("Dense retrieval failed | query=%s", query)
        raise
