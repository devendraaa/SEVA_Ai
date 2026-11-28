
import logging
import os
import time
import cohere
from indexing import load_vectorstores
from retrieval.bm25.retrieval import build_bm25_retriever
from retrieval.hybrid.retrieval import retrieve_hybrid_schemes
from evaluation.retrieval.retrieval_capture import capture_hybrid_retrieval
from evaluation.retrieval.relevance_judge import judge_candidates
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

load_dotenv()

def main():
    try:
        # 1. Initialize Cohere client
        api_key = os.getenv("COHERE_API_KEY")

        if not api_key:
            raise ValueError("COHERE_API_KEY environment variable is missing.")

        cohere_client = cohere.ClientV2(api_key=api_key)

        # 2. Load existing vector stores
        discovery_db, _ = load_vectorstores()

        # 3. Build BM25 index and measure startup latency
        bm25_start = time.perf_counter()

        bm25_retriever = build_bm25_retriever(discovery_db)

        bm25_build_latency = time.perf_counter() - bm25_start

        query = "What government schemes are available for farmers?"

        # 4. Execute complete hybrid retrieval
        result = retrieve_hybrid_schemes(
            query=query,
            discovery_db=discovery_db,
            bm25_retriever=bm25_retriever,
            cohere_client=cohere_client,
            retrieval_k=30,
            rrf_k=60,
            top_n=10,
        )

        # 5. Display retrieval results
        print("\nQuery:", result["query"])
        print("Dense schemes:", result["dense_schemes"])
        print("BM25 schemes:", result["bm25_schemes"])
        print("Fused schemes:", result["fused_schemes"])
        print("Generated profiles:", result["profile_count"])

        print("\nTop 10 Reranked Schemes:")

        for scheme in result["retrieved_schemes"]:
            print(
                f"Rank: {scheme['rank']} | "
                f"Scheme ID: {scheme['scheme_id']} | "
                f"Relevance: {scheme['relevance_score']:.4f} | "
                f"Original RRF Rank: {scheme['original_rrf_rank']}"
            )

        # 6. Display latency
        # latency = result["latency"]

        # print("\nLatency Breakdown:")
        # print(f"BM25 index build : {bm25_build_latency:.4f} sec")
        # print(f"Dense retrieval  : {latency['dense']:.4f} sec")
        # print(f"BM25 retrieval   : {latency['bm25']:.4f} sec")
        # print(f"RRF fusion       : {latency['fusion']:.4f} sec")
        # print(f"Profile building : {latency['scheme_profiles']:.4f} sec")
        # print(f"Cohere reranking : {latency['cohere']:.4f} sec")
        # print(f"Total query time : {latency['total']:.4f} sec")

        # 1. Capture retrieval results
        captured = capture_hybrid_retrieval(
            query=query,
            discovery_db=discovery_db,
            bm25_retriever=bm25_retriever,
            cohere_client=cohere_client,
        )

        # 2. Judge all unique retrieved schemes
        judgments = judge_candidates(
            query=query,
            candidates=captured["candidates"],
        )

        # 3. Connect judgments to their retrieval stages
        judgment_lookup = {
            item["scheme_id"]: item
            for item in judgments
        }

        for candidate in captured["candidates"]:
            candidate["judgment"] = judgment_lookup.get(
                candidate["scheme_id"]
            )

        print(captured["candidates"])

    except Exception:
        logging.exception("Hybrid retrieval test failed")
        raise


if __name__ == "__main__":
    main()
