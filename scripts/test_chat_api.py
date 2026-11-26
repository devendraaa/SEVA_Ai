
import logging
import requests

logger = logging.getLogger(__name__)

API_URL = "http://127.0.0.1:8000/api/v1/chat"

QUERY = "What government schemes are available for farmers in Maharashtra?"


def test_chat_api():
    try:
        response = requests.post(
            API_URL,
            json={"query": QUERY},
            timeout=180,
        )

        response.raise_for_status()
        result = response.json()

        print("\nQUESTION:")
        print(result["query"])

        print("\nANSWER:")
        print(result["answer"])

        print("\nSOURCES:")
        for source in result["sources"]:
            print(
                f"Scheme ID: {source['scheme_id']} | "
                f"Rank: {source['rank']}"
            )

        print("\nRETRIEVAL METHOD:", result["retrieval_method"])
        print("RETRIEVED COUNT:", result["retrieved_count"])

    except requests.RequestException:
        logger.exception("Chat API request failed")
        raise

    except Exception:
        logger.exception("Chat API test failed")
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_chat_api()
