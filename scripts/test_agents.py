import logging

from agents.graphs.graph import run_agent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

def main():
    try:
        query = "Find government schemes for farmers"

        result = run_agent(query)

        print("\n========== SEVA_Ai AGENT ==========")
        print("Query:", result["query"])
        print("Intent:", result["intent"])
        print("Answer:", result["answer"])

    except Exception:
        logger.exception("Agent test failed.")
        raise


if __name__ == "__main__":
    main()