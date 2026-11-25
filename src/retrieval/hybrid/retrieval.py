
import logging

from retrieval.dense.dense import retrieve_dense_schemes
from retrieval.bm25.retrieval import retrieve_bm25_schemes
from retrieval.fusion.rrf import reciprocal_rank_fusion
from retrieval.reranking.scheme_profile import build_scheme_profiles
from retrieval.reranking.cohere_rerank import rerank_schemes

from observability.tracing import trace_span


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

    try:
        # 1. Dense retrieval
        with trace_span(
            "dense-retrieval",
            input_data={"query": query, "k": retrieval_k},
        ) as span:

            dense_result = retrieve_dense_schemes(
                query=query,
                discovery_db=discovery_db,
                k=retrieval_k,
            )

            dense_results = dense_result["retrieved_schemes"]

            if span:
                span.update(
                    output={
                        "count": len(dense_results),
                        "retrieved_schemes": dense_results,
                    }
                )

        # 2. BM25 retrieval
        with trace_span(
            "bm25-retrieval",
            input_data={"query": query},
        ) as span:

            bm25_result = retrieve_bm25_schemes(
                query=query,
                bm25_retriever=bm25_retriever,
            )

            bm25_results = bm25_result["retrieved_schemes"]

            if span:
                span.update(
                    output={
                        "count": len(bm25_results),
                        "retrieved_schemes": bm25_results,
                    }
                )

        # 3. RRF fusion
        with trace_span(
            "rrf-fusion",
            input_data={
                "dense_candidates": dense_results,
                "bm25_candidates": bm25_results,
                "rrf_k": rrf_k,
            },
        ) as span:

            fused_results = reciprocal_rank_fusion(
                dense_results=dense_results,
                bm25_results=bm25_results,
                rrf_k=rrf_k,
            )

            if span:
                span.update(
                    output={
                        "count": len(fused_results),
                        "fused_results": fused_results,
                    }
                )

        # 4. Scheme profile generation
        with trace_span(
            "scheme-profile-generation",
            input_data={
                "fused_scheme_ids": [
                    item["scheme_id"] for item in fused_results
                ]
            },
        ) as span:

            profiles = build_scheme_profiles(
                discovery_db=discovery_db,
                fused_results=fused_results,
            )

            if span:
                span.update(
                    output={
                        "count": len(profiles),
                        "profiles": profiles,
                    }
                )

        # 5. Cohere reranking
        rerank_result = None
        rerank_error = None
        rerank_status = "failed"
        reranked_results = []

        with trace_span(
            "cohere-reranking",
            input_data={
                "query": query,
                "candidate_profiles": profiles,
                "top_n": top_n,
            },
            metadata={"model": "rerank-v5.0-fast"},
        ) as span:

            try:
                rerank_result = rerank_schemes(
                    query=query,
                    candidates=profiles,
                    cohere_client=cohere_client,
                    top_n=top_n,
                )

                reranked_results = rerank_result["reranked_schemes"]
                rerank_status = rerank_result["status"]

                if span:
                    span.update(
                        output={
                            "count": len(reranked_results),
                            "status": rerank_status,
                            "reranked_schemes": reranked_results,
                        }
                    )

            except Exception as exc:
                rerank_error = str(exc)

                logger.exception(
                    "Cohere reranking failed | using RRF fallback"
                )

                if span:
                    span.update(
                        output={
                            "status": "failed",
                            "error": rerank_error,
                        }
                    )

        # 6. Select final results
        if rerank_status == "success" and reranked_results:
            final_results = reranked_results
            final_method = "cohere"

        else:
            final_results = fused_results[:top_n]
            final_method = "rrf_fallback"

        logger.info(
            "Hybrid retrieval completed | "
            "dense=%d | bm25=%d | rrf=%d | "
            "cohere=%d | final=%d | method=%s",
            len(dense_results),
            len(bm25_results),
            len(fused_results),
            len(reranked_results),
            len(final_results),
            final_method,
        )

        return {
            "query": query,
            "dense_results": dense_results,
            "bm25_results": bm25_results,
            "rrf_results": fused_results,
            "scheme_profiles": profiles,
            "reranked_results": reranked_results,
            "retrieved_schemes": final_results,
            "final_method": final_method,
            "rerank_status": rerank_status,
            "rerank_error": rerank_error,
            "dense_schemes": len(dense_results),
            "bm25_schemes": len(bm25_results),
            "fused_schemes": len(fused_results),
            "profile_count": len(profiles),
            "reranked_count": len(reranked_results),
        }

    except Exception:
        logger.exception(
            "Hybrid retrieval pipeline failed | query=%s",
            query,
        )
        raise
