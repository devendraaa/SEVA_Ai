
import logging

from evaluation.datasets.golden_dataset import load_golden_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

logger = logging.getLogger(__name__)


def inspect_ground_truth(dataset_path: str) -> None:
    try:
        df = load_golden_dataset(dataset_path)

        total = len(df)

        doc_ids_present = df["relevant_doc_ids"].apply(
            lambda ids: len(ids) > 0
        )

        scheme_ids_present = df["relevant_scheme_ids"].apply(
            lambda ids: len(ids) > 0
        )

        logger.info("Ground-truth relevance inspection completed.")

        print("\n========== RELEVANCE COVERAGE ==========")

        print(f"Total questions: {total}")

        print(
            f"Questions with relevant_doc_ids: "
            f"{doc_ids_present.sum()}"
        )

        print(
            f"Questions without relevant_doc_ids: "
            f"{(~doc_ids_present).sum()}"
        )

        print(
            f"Questions with relevant_scheme_ids: "
            f"{scheme_ids_present.sum()}"
        )

        print("\n========== DOCUMENT ID DISTRIBUTION ==========")

        print(
            df["relevant_doc_ids"].apply(len).describe()
        )

        print("\n========== SCHEME ID DISTRIBUTION ==========")

        print(
            df["relevant_scheme_ids"].apply(len).describe()
        )

        print("\n========== NON-EMPTY DOCUMENT ID SAMPLES ==========")

        samples = df[doc_ids_present].head(5)

        if samples.empty:
            print("No relevant_doc_ids found.")
        else:
            for _, row in samples.iterrows():
                print(f"\nTest ID: {row['test_id']}")
                print(f"Question: {row['question']}")
                print(f"Document IDs: {row['relevant_doc_ids']}")
                print(f"Scheme IDs: {row['relevant_scheme_ids']}")

        print("\n========== DATASET COLUMNS ==========")
        print(df.columns.tolist())

    except Exception:
        logger.exception("Ground-truth inspection failed.")
        raise


if __name__ == "__main__":
    try:
        inspect_ground_truth(
            "notebooks/golden_dataset/final/Final_dataset_splits/test.json"
        )

    except Exception:
        logger.exception("Inspection script failed.")
        raise
