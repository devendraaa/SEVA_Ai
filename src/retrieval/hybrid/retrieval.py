
import logging
import time
from retrieval.dense.dense import retrieve_dense_schemes
from retrieval.bm25.retrieval import retrieve_bm25_schemes
from retrieval.fusion.rrf import reciprocal_rank_fusion
from retrieval.reranking.scheme_profile import build_scheme_profiles
from retrieval.reranking.cohere_rerank import rerank_schemes

logger = logging.getLogger(__name__)


def retrieve_hybrid_schemes(
    query: str,
    discovery_db,
    bm25_retriever,
    cohere_client,
    retrieval_k: int = 30,
    rrf_k: int = 60,
    top_n: int = 10,
) -> dict:
    """Execute Dense + BM25 + RRF + Cohere retrieval."""

    pipeline_start = time.perf_counter()

    try:
        # 1. Dense retrieval
        dense_result = retrieve_dense_schemes(
            query=query,
            discovery_db=discovery_db,
            k=retrieval_k,
        )

        # 2. BM25 retrieval
        bm25_result = retrieve_bm25_schemes(
            query=query,
            bm25_retriever=bm25_retriever,
        )

        # 3. RRF fusion
        fusion_start = time.perf_counter()

        fused_results = reciprocal_rank_fusion(
            dense_results=dense_result["retrieved_schemes"],
            bm25_results=bm25_result["retrieved_schemes"],
            rrf_k=rrf_k,
        )

        fusion_latency = time.perf_counter() - fusion_start

        # 4. Scheme profile generation
        profile_start = time.perf_counter()

        profiles = build_scheme_profiles(
            discovery_db=discovery_db,
            fused_results=fused_results,
        )

        profile_latency = time.perf_counter() - profile_start

        # 5. Cohere reranking
        rerank_result = rerank_schemes(
            query=query,
            candidates=profiles,
            cohere_client=cohere_client,
            top_n=top_n,
        )

        total_latency = time.perf_counter() - pipeline_start

        latency_breakdown = {
            "dense": dense_result["latency"],
            "bm25": bm25_result["latency"],
            "fusion": fusion_latency,
            "scheme_profiles": profile_latency,
            "cohere": rerank_result["latency"],
            "total": total_latency,
        }

        logger.info(
            "Hybrid retrieval completed | "
            "dense=%d | bm25=%d | fused=%d | profiles=%d | "
            "reranked=%d | total_latency=%.4fs",
            len(dense_result["retrieved_schemes"]),
            len(bm25_result["retrieved_schemes"]),
            len(fused_results),
            len(profiles),
            len(rerank_result["reranked_schemes"]),
            total_latency,
        )

        return {
            "query": query,
            "dense_schemes": len(dense_result["retrieved_schemes"]),
            "bm25_schemes": len(bm25_result["retrieved_schemes"]),
            "fused_schemes": len(fused_results),
            "profile_count": len(profiles),
            "retrieved_schemes": rerank_result["reranked_schemes"],
            "latency": latency_breakdown,
        }

    except Exception:
        logger.exception(
            "Hybrid retrieval pipeline failed | query=%s",
            query,
        )
        raise
