
import logging

from dotenv import load_dotenv

from evaluation.datasets.golden_dataset import (
    load_golden_dataset,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)


def main():
    try:
        load_dotenv()

        dataset_path = (
            "G:\\SEVA_Ai\\SEVA_Ai\\notebooks\\golden_dataset\\discovery\\discovery_131_validation_input.jsonl"
        )

        df = load_golden_dataset(dataset_path)

        print("\nDataset shape:", df.shape)

        print("\nUnique schemes:", df["scheme_id"].nunique())

        print("\nQuestions per scheme:")
        print(
            df.groupby("scheme_name")
            .size()
            .sort_values(ascending=False)
        )

        print("\nSample records:")
        print(
            df[
                [
                    "test_id",
                    "scheme_name",
                    "question",
                    "relevant_scheme_ids",
                ]
            ].head()
        )

    except Exception:
        logger.exception(
            "Golden dataset test failed."
        )
        raise


if __name__ == "__main__":
    main()
