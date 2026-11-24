
import logging

from evaluation.datasets.golden_dataset import load_golden_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

logger = logging.getLogger(__name__)


def main():
    try:
        dataset_path = "G:\\SEVA_Ai\\SEVA_Ai\\notebooks\\golden_dataset\\final\\Final_dataset_splits\\test.json"

        golden_df = load_golden_dataset(dataset_path)

        print("\nDataset shape:", golden_df.shape)

        print("\nFirst question:")
        print(golden_df.iloc[0]["question"])

        print("\nRelevant document IDs:")
        print(golden_df.iloc[0]["relevant_doc_ids"])

        print("\nRelevant scheme IDs:")
        print(golden_df.iloc[0]["relevant_scheme_ids"])

        print("\nAnswerability distribution:")
        print(golden_df["answerable"].value_counts())

    except Exception:
        logger.exception("Evaluation dataset test failed.")
        raise


if __name__ == "__main__":
    main()
