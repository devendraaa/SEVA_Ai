
import logging
import time
import threading

logger = logging.getLogger(__name__)

COHERE_RERANK_MODEL = "rerank-v5.0-fast"

# Rate-limit configuration
REQUEST_INTERVAL = 10
MAX_RETRIES = 8

# Shared request timing across calls in this Python process
_last_request_time = 0.0
_request_lock = threading.Lock()


def _get_http_status(exception: Exception) -> int | None:
    """Extract HTTP status code from a Cohere exception."""

    status = getattr(exception, "status_code", None)

    if status is None:
        status = getattr(
            getattr(exception, "response", None),
            "status_code",
            None,
        )

    return status


def rerank_schemes(
    query: str,
    candidates: list[dict],
    cohere_client,
    top_n: int = 10,
) -> dict:
    """
    Rerank scheme profiles using Cohere.

    Features:
    - Request interval control
    - HTTP 429 retry handling
    - Exponential backoff for server errors
    - Retry tracking
    - API latency tracking
    """

    global _last_request_time

    total_start = time.perf_counter()

    try:
        # --------------------------------------------------
        # 1. Validate candidates
        # --------------------------------------------------
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
                "retry_count": 0,
                "status": "skipped",
                "error": None,
            }

        if top_n < 1:
            raise ValueError("top_n must be greater than 0")

        # --------------------------------------------------
        # 2. Prepare documents
        # --------------------------------------------------
        documents = [
            candidate["text"]
            for candidate in candidates
        ]

        effective_top_n = min(top_n, len(documents))

        retries = 0
        api_latency = 0.0

        # --------------------------------------------------
        # 3. Cohere API request with rate limiting
        # --------------------------------------------------
        with _request_lock:

            while True:

                # Respect minimum request interval
                elapsed = time.monotonic() - _last_request_time

                wait_time = REQUEST_INTERVAL - elapsed

                if wait_time > 0:
                    logger.info(
                        "Cohere rate limiter | waiting %.2fs",
                        wait_time,
                    )

                    time.sleep(wait_time)

                request_start = time.perf_counter()

                try:
                    response = cohere_client.rerank(
                        model=COHERE_RERANK_MODEL,
                        query=query,
                        documents=documents,
                        top_n=effective_top_n,
                    )

                    api_latency = (
                        time.perf_counter() - request_start
                    )

                    _last_request_time = time.monotonic()

                    break

                except Exception as exc:

                    # Record the time of every attempted request
                    _last_request_time = time.monotonic()

                    status = _get_http_status(exc)

                    if status == 429:
                        retries += 1
                        wait = 65

                    elif status in (500, 502, 503, 504):
                        retries += 1

                        wait = min(
                            60,
                            5 * (2 ** (retries - 1)),
                        )

                    else:
                        raise

                    if retries > MAX_RETRIES:
                        logger.error(
                            "Cohere maximum retries exceeded | "
                            "status=%s | retries=%d | query=%s",
                            status,
                            retries - 1,
                            query,
                        )

                        raise

                    logger.warning(
                        "Cohere API retry | "
                        "HTTP %s | retry=%d/%d | "
                        "waiting=%ds | query=%s",
                        status,
                        retries,
                        MAX_RETRIES,
                        wait,
                        query,
                    )

                    time.sleep(wait)

        # --------------------------------------------------
        # 4. Process reranking results
        # --------------------------------------------------
        reranked_schemes = []

        for rank, result in enumerate(
            response.results,
            start=1,
        ):
            candidate = candidates[result.index]

            reranked_schemes.append({
                "scheme_id": candidate["scheme_id"],
                "rank": rank,
                "relevance_score": float(
                    result.relevance_score
                ),
                "original_rrf_rank": candidate["rank"],
                "rrf_score": candidate["rrf_score"],
            })

        total_latency = time.perf_counter() - total_start

        # --------------------------------------------------
        # 5. Logging
        # --------------------------------------------------
        logger.info(
            "Cohere reranking completed | "
            "candidates=%d | returned=%d | "
            "api_latency=%.4fs | total_latency=%.4fs | "
            "retries=%d",
            len(candidates),
            len(reranked_schemes),
            api_latency,
            total_latency,
            retries,
        )

        # --------------------------------------------------
        # 6. Return results
        # --------------------------------------------------
        return {
            "query": query,
            "model": COHERE_RERANK_MODEL,
            "candidates": len(candidates),
            "reranked_schemes": reranked_schemes,
            "latency": api_latency,
            "total_latency": total_latency,
            "retry_count": retries,
            "status": "success",
            "error": None,
        }

    except Exception as exc:
        logger.exception(
            "Cohere reranking failed | query=%s",
            query,
        )

        raise
