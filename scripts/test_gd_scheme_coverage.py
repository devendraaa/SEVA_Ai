
import logging

from dotenv import load_dotenv
from indexing import load_vectorstores

from evaluation.datasets.golden_dataset import load_golden_dataset

# Import your existing Discovery DB initialization function here.
# Use the same initialization you use in your hybrid evaluator.

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


def main():
    try:
        load_dotenv()

        dataset_path = (
            "notebooks/golden_dataset/discovery/"
            "discovery_131_validation_input.jsonl"
        )

        df = load_golden_dataset(dataset_path)

        expected_ids = set(df["scheme_id"])

        # TODO: Initialize your existing discovery_db here.
        # Do not create a new vector store.
        discovery_db, _ = load_vectorstores()

        # Replace this with your existing Chroma collection.
        collection = discovery_db._collection

        indexed_ids = set()

        for offset in range(0, collection.count(), 1000):
            batch = collection.get(
                limit=1000,
                offset=offset,
                include=["metadatas"],
            )

            for metadata in batch["metadatas"]:
                if metadata and metadata.get("scheme_id"):
                    indexed_ids.add(metadata["scheme_id"])

        found_ids = expected_ids.intersection(indexed_ids)
        missing_ids = expected_ids - indexed_ids

        logger.info("Golden schemes: %s", len(expected_ids))
        logger.info("Indexed schemes: %s", len(indexed_ids))
        logger.info("Found schemes: %s", len(found_ids))
        logger.info("Missing schemes: %s", len(missing_ids))

        if missing_ids:
            logger.warning("Missing scheme IDs: %s", sorted(missing_ids))

    except Exception:
        logger.exception("Scheme coverage validation failed.")
        raise


if __name__ == "__main__":
    main()
