
import logging
import os
import cohere
from dotenv import load_dotenv
from indexing import load_vectorstores
from retrieval.bm25.retrieval import build_bm25_retriever
from evaluation.retrieval.retrieval_capture import capture_hybrid_retrieval
from evaluation.retrieval.relevance_judge import judge_candidates
from evaluation.retrieval.save_result import save_analysis

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


def main():
    try:
        load_dotenv()

        api_key = os.getenv("COHERE_API_KEY")
        if not api_key:
            raise ValueError("COHERE_API_KEY is missing.")

        cohere_client = cohere.ClientV2(api_key=api_key)

        discovery_db, _ = load_vectorstores()
        bm25_retriever = build_bm25_retriever(discovery_db)

        query = "What government schemes are available for farmers?"

        # 1. Retrieve and capture all stages.
        captured = capture_hybrid_retrieval(
            query=query,
            discovery_db=discovery_db,
            bm25_retriever=bm25_retriever,
            cohere_client=cohere_client,
        )

        # 2. Judge unique retrieved scheme profiles.
        judgments = judge_candidates(
            query=query,
            candidates=captured["candidates"],
        )

        judgment_lookup = {
            item["scheme_id"]: item
            for item in judgments
        }

        # 3. Display results.
        print(f"\nQuery: {query}")
        print(f"Final method: {captured['final_method']}")
        print(f"Cohere status: {captured['rerank_status']}")

        print("\n========== RELEVANCE ANALYSIS ==========")

        for candidate in captured["candidates"]:
            judgment = judgment_lookup.get(candidate["scheme_id"])

            print(f"\nScheme ID: {candidate['scheme_id']}")
            print(f"Stage ranks: {candidate['stage_ranks']}")

            if judgment:
                print(f"Verdict: {judgment['verdict']}")
                print(f"Score: {judgment['relevance_score']}")
                print(f"Reason: {judgment['reason']}")
            else:
                print("Verdict: UNJUDGED")

        judgments = judge_candidates(
            query=query,
            candidates=captured["candidates"],
            )

        save_analysis(captured, judgments)
        

    except Exception:
        logger.exception("Hybrid retrieval test failed")
        raise


if __name__ == "__main__":
    main()
