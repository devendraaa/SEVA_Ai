
import logging

logger = logging.getLogger(__name__)

def reciprocal_rank_fusion(
    dense_results: list[dict],
    bm25_results: list[dict],
    rrf_k: int = 60,
) -> list[dict]:
    """Combine Dense and BM25 rankings using RRF."""

    try:
        scores = {}
        first_seen = {}

        rankings = [dense_results, bm25_results]

        for ranking in rankings:
            for item in ranking:
                scheme_id = item["scheme_id"]
                rank = item["rank"]

                if scheme_id not in scores:
                    scores[scheme_id] = 0.0
                    first_seen[scheme_id] = len(first_seen)

                scores[scheme_id] += 1 / (rrf_k + rank)

        fused_results = sorted(
            scores.items(),
            key=lambda item: (
                -item[1],
                first_seen[item[0]],
            ),
        )

        results = [
            {
                "scheme_id": scheme_id,
                "rrf_score": score,
                "rank": rank,
            }
            for rank, (scheme_id, score) in enumerate(
                fused_results,
                start=1,
            )
        ]

        logger.info(
            "RRF fusion completed | dense=%d | bm25=%d | fused=%d",
            len(dense_results),
            len(bm25_results),
            len(results),
        )

        return results

    except Exception:
        logger.exception("Reciprocal Rank Fusion failed")
        raise
