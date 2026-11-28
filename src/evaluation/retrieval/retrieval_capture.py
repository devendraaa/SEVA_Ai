import logging
import time

from retrieval.hybrid.retrieval import retrieve_hybrid_schemes
from retrieval.reranking.scheme_profile import build_scheme_profiles

logger = logging.getLogger(__name__)

def capture_hybrid_retrieval(
    query: str,
    discovery_db,
    bm25_retriever,
    cohere_client,
    retrieval_k: int = 30,
    rrf_k: int = 60,
    top_n: int = 10,
) -> dict:
    """Capture results from Dense, BM25, RRF and Cohere."""

    start_time = time.perf_counter()

    try:
        # Run the existing hybrid retrieval pipeline.
        result = retrieve_hybrid_schemes(
            query=query,
            discovery_db=discovery_db,
            bm25_retriever=bm25_retriever,
            cohere_client=cohere_client,
            retrieval_k=retrieval_k,
            rrf_k=rrf_k,
            top_n=top_n,
        )

        stages = {
            "dense": result["dense_results"],
            "bm25": result["bm25_results"],
            "rrf": result["rrf_results"],
            "cohere": result["reranked_results"],
        }

        # Collect unique scheme IDs from every stage.
        candidates = {}

        for stage_name, schemes in stages.items():
            for scheme in schemes:
                scheme_id = scheme.get("scheme_id")

                if not scheme_id:
                    continue

                if scheme_id not in candidates:
                    candidates[scheme_id] = {
                        "scheme_id": scheme_id,
                        "stage_ranks": {},
                    }

                candidates[scheme_id]["stage_ranks"][stage_name] = (
                    scheme.get("rank")
                )

        # Build profiles for all unique candidates.
        rrf_lookup = {
            item["scheme_id"]: item
            for item in result["rrf_results"]
        }

        profile_candidates = []

        for index, scheme_id in enumerate(candidates, start=1):
            rrf_item = rrf_lookup.get(scheme_id, {})

            profile_candidates.append({
                "scheme_id": scheme_id,
                "rrf_score": rrf_item.get("rrf_score", 0.0),
                "rank": rrf_item.get("rank", index),
            })

        profiles = build_scheme_profiles(
            discovery_db=discovery_db,
            fused_results=profile_candidates,
        )

        # Attach profile text to each candidate.
        profile_lookup = {
            item["scheme_id"]: item["text"]
            for item in profiles
        }

        captured_candidates = []

        for scheme_id, candidate in candidates.items():
            text = profile_lookup.get(scheme_id)

            if not text:
                continue

            captured_candidates.append({
                "scheme_id": scheme_id,
                "stage_ranks": candidate["stage_ranks"],
                "text": text,
            })

        latency = time.perf_counter() - start_time

        logger.info(
            "Retrieval capture completed | candidates=%d | latency=%.2fs",
            len(captured_candidates),
            latency,
        )

        return {
            "query": query,
            "stages": stages,
            "candidates": captured_candidates,
            "retrieved_schemes": result["retrieved_schemes"],
            "final_method": result["final_method"],
            "rerank_status": result["rerank_status"],
            "latency": latency,
        }

    except Exception:
        logger.exception("Retrieval capture failed | query=%s", query)
        raise
