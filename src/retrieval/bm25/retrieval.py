
import logging
import re
import time

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

BM25_K = 30


def bm25_tokenize(text: str) -> list[str]:
    """Tokenize text for BM25."""
    return re.findall(r"\w+", text.lower())


def build_bm25_retriever(discovery_db) -> BM25Retriever:
    """Build an in-memory BM25 index from existing Chroma documents."""

    start_time = time.perf_counter()

    try:
        data = discovery_db.get(
            include=["documents", "metadatas"]
        )

        documents = [
            Document(
                page_content=content,
                metadata=metadata or {},
            )
            for content, metadata in zip(
                data["documents"],
                data["metadatas"],
            )
            if content and metadata
        ]

        retriever = BM25Retriever.from_documents(
            documents,
            k=BM25_K,
            preprocess_func=bm25_tokenize,
        )

        latency = time.perf_counter() - start_time

        logger.info(
            "BM25 index built | documents=%d | latency=%.4fs",
            len(documents),
            latency,
        )

        return retriever

    except Exception:
        logger.exception("BM25 index construction failed")
        raise


def retrieve_bm25_schemes(
    query: str,
    bm25_retriever: BM25Retriever,
) -> dict:
    """Retrieve schemes using BM25."""

    start_time = time.perf_counter()

    try:
        results = bm25_retriever.invoke(query)

        retrieved_schemes = []
        seen_scheme_ids = set()

        for chunk_rank, doc in enumerate(results, start=1):
            scheme_id = doc.metadata.get("scheme_id")

            if scheme_id and scheme_id not in seen_scheme_ids:
                seen_scheme_ids.add(scheme_id)

                retrieved_schemes.append({
                    "scheme_id": scheme_id,
                    "rank": len(retrieved_schemes) + 1,
                    "chunk_rank": chunk_rank,
                    "section": doc.metadata.get("child_type"),
                })

        latency = time.perf_counter() - start_time

        logger.info(
            "BM25 retrieval completed | chunks=%d | schemes=%d | latency=%.4fs",
            len(results),
            len(retrieved_schemes),
            latency,
        )

        return {
            "query": query,
            "retrieved_chunks": len(results),
            "retrieved_schemes": retrieved_schemes,
            "latency": latency,
        }

    except Exception:
        logger.exception("BM25 retrieval failed | query=%s", query)
        raise
