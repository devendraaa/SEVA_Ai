
import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def load_json_records(path: Path) -> list[dict]:
    """Load records from standard JSON or JSONL format."""

    try:
        content = path.read_text(encoding="utf-8").strip()

        if not content:
            raise ValueError("Golden dataset file is empty.")

        # First, try standard JSON format
        try:
            records = json.loads(content)

            if isinstance(records, dict):
                records = [records]

            if not isinstance(records, list):
                raise ValueError(
                    "JSON dataset must contain a list of records."
                )

            logger.info("Detected standard JSON format.")
            return records

        except json.JSONDecodeError:
            # If standard JSON fails, try JSONL
            logger.warning(
                "Standard JSON parsing failed. "
                "Trying JSONL format."
            )

            records = []

            for line_number, line in enumerate(
                content.splitlines(), start=1
            ):
                if not line.strip():
                    continue

                try:
                    record = json.loads(line)
                    records.append(record)

                except json.JSONDecodeError:
                    logger.exception(
                        "Invalid JSONL record at line %s.",
                        line_number,
                    )
                    raise

            logger.info("Detected JSONL format.")
            return records

    except Exception:
        logger.exception(
            "Failed to read golden dataset file: %s", path
        )
        raise


def load_golden_dataset(file_path: str) -> pd.DataFrame:
    """
    Load and validate the scheme-specific golden dataset.

    Supports:
    - JSON: a list of JSON objects
    - JSONL: one JSON object per line
    """

    try:
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Golden dataset not found: {path}"
            )

        logger.info("Loading golden dataset: %s", path)

        records = load_json_records(path)

        if not records:
            raise ValueError("Golden dataset contains no records.")

        required_fields = {
            "scheme_id",
            "scheme_name",
            "question_id",
            "question",
            "scheme_context",
        }

        validated_records = []
        invalid_records = []

        for index, record in enumerate(records):

            try:
                if not isinstance(record, dict):
                    raise ValueError(
                        "Each dataset record must be a dictionary."
                    )

                missing_fields = required_fields - record.keys()

                if missing_fields:
                    raise ValueError(
                        f"Missing fields: {sorted(missing_fields)}"
                    )

                scheme_id = str(record["scheme_id"]).strip()
                scheme_name = str(record["scheme_name"]).strip()
                question = str(record["question"]).strip()
                question_id = record["question_id"]
                scheme_context = record["scheme_context"]

                if not scheme_id or not scheme_name or not question:
                    raise ValueError(
                        "scheme_id, scheme_name and question "
                        "cannot be empty."
                    )

                if question_id is None or not str(question_id).strip():
                    raise ValueError("question_id cannot be empty.")

                if not isinstance(scheme_context, dict):
                    raise ValueError(
                        "scheme_context must be a dictionary."
                    )

                context_scheme_id = str(
                    scheme_context.get("scheme_id", "")
                ).strip()

                if context_scheme_id != scheme_id:
                    raise ValueError(
                        "scheme_id does not match "
                        "scheme_context.scheme_id."
                    )

                context_scheme_name = str(
                    scheme_context.get("name", "")
                ).strip()

                if context_scheme_name != scheme_name:
                    raise ValueError(
                        "scheme_name does not match "
                        "scheme_context.name."
                    )

                test_id = f"{scheme_id}_{question_id}"

                validated_records.append({
                    "test_id": test_id,
                    "scheme_id": scheme_id,
                    "scheme_name": scheme_name,
                    "question_id": question_id,
                    "question": question,
                    "relevant_scheme_ids": [scheme_id],
                    "scheme_context": scheme_context,
                })

            except Exception as exc:
                invalid_records.append({
                    "record_index": index,
                    "error": str(exc),
                })

                logger.warning(
                    "Invalid record | index=%s | error=%s",
                    index,
                    exc,
                )

        if invalid_records:
            logger.warning(
                "Found %s invalid records out of %s.",
                len(invalid_records),
                len(records),
            )

        if not validated_records:
            raise ValueError(
                "No valid records found in golden dataset."
            )

        df = pd.DataFrame(validated_records)

        # Validate unique test IDs
        duplicate_ids = df[
            df["test_id"].duplicated(keep=False)
        ]

        if not duplicate_ids.empty:
            duplicate_values = (
                duplicate_ids["test_id"].tolist()
            )

            raise ValueError(
                f"Duplicate test IDs found: {duplicate_values}"
            )

        # Identify duplicate questions
        duplicate_questions = df[
            df["question"].duplicated(keep=False)
        ]

        # Dataset statistics
        unique_schemes = df["scheme_id"].nunique()

        logger.info("Golden dataset loaded successfully.")
        logger.info("Total records: %s", len(records))
        logger.info("Valid records: %s", len(df))
        logger.info("Invalid records: %s", len(invalid_records))
        logger.info("Unique schemes: %s", unique_schemes)
        logger.info(
            "Duplicate questions: %s",
            len(duplicate_questions),
        )

        return df

    except Exception:
        logger.exception(
            "Failed to load golden dataset: %s",
            file_path,
        )
        raise
