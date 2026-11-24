
import logging
from typing import Sequence
import math

logger = logging.getLogger(__name__)


def calculate_retrieval_metrics(
    retrieved_ids: Sequence[str],
    relevant_ids: Sequence[str],
    k_values: Sequence[int] = (1, 3, 5, 10),
) -> dict:
    """
    Calculate Recall@K, MRR, MAP and NDCG@K.

    Uses binary relevance and deduplicated ranked identifiers.
    """

    try:
        if not relevant_ids:
            raise ValueError("Ground-truth relevant IDs cannot be empty.")

        if not retrieved_ids:
            logger.warning("No documents retrieved for this query.")

        relevant_set = set(relevant_ids)

        # Preserve ranking order and remove duplicate IDs.
        ranked_ids = list(dict.fromkeys(retrieved_ids))

        reciprocal_rank = 0.0
        average_precision = 0.0
        relevant_found = 0

        for rank, doc_id in enumerate(ranked_ids, start=1):
            if doc_id in relevant_set:
                relevant_found += 1

                if reciprocal_rank == 0:
                    reciprocal_rank = 1.0 / rank

                average_precision += relevant_found / rank

        average_precision /= len(relevant_set)

        metrics = {
            "mrr": reciprocal_rank,
            "map": average_precision,
        }

        for k in k_values:
            if k <= 0:
                raise ValueError(f"K must be positive. Received: {k}")

            top_k = ranked_ids[:k]

            relevant_retrieved = sum(
                doc_id in relevant_set for doc_id in top_k
            )

            recall = relevant_retrieved / len(relevant_set)

            # Binary relevance: IDCG is based on the number
            # of relevant documents available, up to K.
            dcg = sum(
                1.0 / math.log2(rank + 1)
                for rank, doc_id in enumerate(top_k, start=1)
                if doc_id in relevant_set
            )

            ideal_relevant_count = min(len(relevant_set), k)

            idcg = sum(
                1.0 / math.log2(rank + 1)
                for rank in range(1, ideal_relevant_count + 1)
            )

            ndcg = dcg / idcg if idcg > 0 else 0.0

            metrics[f"recall@{k}"] = recall
            metrics[f"ndcg@{k}"] = ndcg

        logger.debug(
            "Retrieval metrics calculated | retrieved=%d | relevant=%d",
            len(ranked_ids),
            len(relevant_set),
        )

        return metrics

    except Exception:
        logger.exception("Retrieval metric calculation failed.")
        raise
