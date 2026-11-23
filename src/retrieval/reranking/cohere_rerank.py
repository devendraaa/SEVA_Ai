
import logging
import time

logger = logging.getLogger(__name__)

COHERE_RERANK_MODEL = "rerank-v5.0-fast"


def rerank_schemes(
    query: str,
    candidates: list[dict],
    cohere_client,
    top_n: int = 10,
) -> dict:
    """Rerank scheme profiles using Cohere."""

    start_time = time.perf_counter()

    try:
        if not candidates:
            logger.info(
                "Cohere reranking skipped | no candidates | query=%s",
                query,
            )

            return {
                "query": query,
                "model": COHERE_RERANK_MODEL,
                "candidates": 0,
                "reranked_schemes": [],
                "latency": 0.0,
            }

        documents = [
            candidate["text"]
            for candidate in candidates
        ]

        response = cohere_client.rerank(
            model=COHERE_RERANK_MODEL,
            query=query,
            documents=documents,
            top_n=min(top_n, len(documents)),
        )

        reranked_schemes = []

        for rank, result in enumerate(
            response.results,
            start=1,
        ):
            candidate = candidates[result.index]

            reranked_schemes.append({
                "scheme_id": candidate["scheme_id"],
                "rank": rank,
                "relevance_score": float(result.relevance_score),
                "original_rrf_rank": candidate["rank"],
                "rrf_score": candidate["rrf_score"],
            })

        latency = time.perf_counter() - start_time

        logger.info(
            "Cohere reranking completed | candidates=%d | returned=%d | latency=%.4fs",
            len(candidates),
            len(reranked_schemes),
            latency,
        )

        return {
            "query": query,
            "model": COHERE_RERANK_MODEL,
            "candidates": len(candidates),
            "reranked_schemes": reranked_schemes,
            "latency": latency,
        }

    except Exception:
        logger.exception("Cohere reranking failed | query=%s", query)
        raise
