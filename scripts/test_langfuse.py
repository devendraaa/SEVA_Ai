
import logging
from dotenv import load_dotenv
from observability.langfuse_client import get_langfuse_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)

def test_langfuse_connection():
    try:
        load_dotenv()

        langfuse = get_langfuse_client()

        with langfuse.start_as_current_observation(
            as_type="span",
            name="seva-ai-connectivity-test",
            input={
                "project": "SEVA_Ai",
                "test": "Langfuse Cloud connectivity",
            },
        ) as span:
            span.update(
                output={
                    "status": "success",
                    "message": "First SEVA_Ai trace received.",
                }
            )

        langfuse.flush()

        logger.info("Langfuse test trace submitted successfully.")

    except Exception:
        logger.exception("Langfuse connectivity test failed.")
        raise


if __name__ == "__main__":
    test_langfuse_connection()
