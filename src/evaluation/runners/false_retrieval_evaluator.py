
import argparse
import json
import logging
import os
import time

from datetime import datetime
from pathlib import Path

import cohere
import ir_measures
import pandas as pd

from dotenv import load_dotenv
from ir_measures import AP, RR, nDCG, P, R, Qrel, ScoredDoc

from indexing import load_vectorstores
from retrieval.bm25.retrieval import build_bm25_retriever
from retrieval.hybrid.retrieval import retrieve_hybrid_schemes

from evaluation.retrieval.relevance_judge import judge_candidates

from observability.langfuse_client import get_langfuse_client
from observability.tracing import trace_span


logger = logging.getLogger(__name__)

DATASET_PATH = (
    "notebooks/golden_dataset/final/"
    "Final_dataset_splits/test.json"
)

OUTPUT_DIR = "artifacts/evaluation/false_retrieval"

K_VALUES = (5, 10)

IR_METRICS = [
    RR,
    AP,
    nDCG @ 10,
    P @ 5,
    P @ 10,
    R @ 10,
]


def load_dataset(path):
    """Load JSON or JSONL golden dataset."""

    try:
        content = Path(path).read_text(encoding="utf-8").strip()

        if not content:
            raise ValueError("Dataset is empty.")

        try:
            records = json.loads(content)

            if isinstance(records, dict):
                records = records.get("data", records.get("records", [records]))

        except json.JSONDecodeError:
            records = [
                json.loads(line)
                for line in content.splitlines()
                if line.strip()
            ]

        if not isinstance(records, list):
            raise ValueError("Dataset must contain a list of records.")

        rows = []

        for index, record in enumerate(records):
            if not record.get("answerable", True):
                continue

            question = str(record.get("question", "")).strip()

            relevant_ids = record.get("relevant_scheme_ids")

            if relevant_ids is None:
                relevant_ids = [record.get("scheme_id")]

            relevant_ids = list(dict.fromkeys(
                str(item) for item in relevant_ids if item
            ))

            if not question or not relevant_ids:
                logger.warning(
                    "Skipping record without question or relevant IDs | index=%d",
                    index,
                )
                continue

            rows.append({
                "test_id": str(
                    record.get("test_id")
                    or record.get("question_id")
                    or index
                ),
                "question": question,
                "relevant_scheme_ids": relevant_ids,
                "ground_truth_answer": record.get(
                    "ground_truth_answer", ""
                ),
            })

        if not rows:
            raise ValueError("No answerable records with relevant IDs.")

        logger.info("Dataset loaded | questions=%d", len(rows))
        return rows

    except Exception:
        logger.exception("Dataset loading failed | path=%s", path)
        raise


def extract_ids(results):
    return list(dict.fromkeys(
        str(item["scheme_id"])
        for item in results
        if item.get("scheme_id")
    ))


def calculate_id_errors(retrieved_ids, relevant_ids):
    """Compare retrieved scheme IDs against golden labels."""

    retrieved = set(retrieved_ids)
    relevant = set(relevant_ids)

    tp = retrieved & relevant
    fp = retrieved - relevant
    fn = relevant - retrieved

    precision = len(tp) / len(retrieved) if retrieved else 0.0
    recall = len(tp) / len(relevant) if relevant else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    return {
        "tp": len(tp),
        "fp": len(fp),
        "fn": len(fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_ids": sorted(fp),
        "false_negative_ids": sorted(fn),
    }


def calculate_ir_metrics(query_id, retrieved_ids, relevant_ids):
    """Calculate standard ranking metrics with ir-measures."""

    qrels = [
        Qrel(query_id, scheme_id, 1)
        for scheme_id in relevant_ids
    ]

    run = [
        ScoredDoc(query_id, scheme_id, 1 / rank)
        for rank, scheme_id in enumerate(retrieved_ids, start=1)
    ]

    values = ir_measures.calc_aggregate(
        IR_METRICS,
        qrels,
        run,
    )

    return {
        str(metric): float(value)
        for metric, value in values.items()
    }


def calculate_semantic_metrics(
    query_id,
    retrieved_ids,
    judgment_lookup,
):
    """Evaluate only candidates that received a semantic judgment."""

    judged_ids = [
        scheme_id
        for scheme_id in retrieved_ids
        if scheme_id in judgment_lookup
    ]

    qrels = []

    for scheme_id, judgment in judgment_lookup.items():
        verdict = judgment["verdict"]

        grade = {
            "irrelevant": 0,
            "partially_relevant": 1,
            "relevant": 2,
        }[verdict]

        qrels.append(Qrel(query_id, scheme_id, grade))

    run = [
        ScoredDoc(query_id, scheme_id, 1 / rank)
        for rank, scheme_id in enumerate(judged_ids, start=1)
    ]

    if not qrels:
        return {
            "semantic_judged": 0,
            "semantic_tp": 0,
            "semantic_fp": 0,
            "semantic_precision": None,
            "semantic_fp_ids": [],
            "semantic_unjudged_ids": retrieved_ids,
            "semantic_ir_metrics": {},
        }

    semantic_fp = [
        scheme_id
        for scheme_id in judged_ids
        if judgment_lookup[scheme_id]["verdict"] == "irrelevant"
    ]

    semantic_tp = [
        scheme_id
        for scheme_id in judged_ids
        if judgment_lookup[scheme_id]["verdict"] != "irrelevant"
    ]

    semantic_precision = (
        len(semantic_tp) / len(judged_ids)
        if judged_ids else 0.0
    )

    values = ir_measures.calc_aggregate(
        IR_METRICS,
        qrels,
        run,
    )

    return {
        "semantic_judged": len(judged_ids),
        "semantic_tp": len(semantic_tp),
        "semantic_fp": len(semantic_fp),
        "semantic_precision": semantic_precision,
        "semantic_fp_ids": semantic_fp,
        "semantic_unjudged_ids": [
            scheme_id
            for scheme_id in retrieved_ids
            if scheme_id not in judgment_lookup
        ],
        "semantic_ir_metrics": {
            str(metric): float(value)
            for metric, value in values.items()
        },
    }


def log_langfuse_scores(langfuse, query_id, stage, metrics):
    try:
        for name, value in metrics.items():
            if isinstance(value, (int, float)) and value is not None:
                langfuse.score_current_trace(
                    name=f"{stage}_{name}",
                    value=float(value),
                    data_type="NUMERIC",
                    metadata={"test_id": query_id},
                )

    except Exception:
        logger.exception("Langfuse score logging failed.")


def evaluate_query(
    row,
    discovery_db,
    bm25_retriever,
    cohere_client,
    langfuse,
    run_id,
):
    """Run retrieval, semantic judging and metric calculation."""

    query = row["question"]
    query_id = row["test_id"]
    relevant_ids = row["relevant_scheme_ids"]

    try:
        with trace_span(
            name="false-retrieval-evaluation",
            input_data={
                "test_id": query_id,
                "question": query,
                "relevant_scheme_ids": relevant_ids,
            },
            metadata={
                "run_id": run_id,
                "retrieval": "dense_bm25_rrf_cohere",
                "retrieval_k": 30,
                "rrf_k": 60,
                "top_n": 10,
                "judge": "existing_llm_fallback",
            },
        ) as trace:

            start = time.perf_counter()

            result = retrieve_hybrid_schemes(
                query=query,
                discovery_db=discovery_db,
                bm25_retriever=bm25_retriever,
                cohere_client=cohere_client,
                retrieval_k=30,
                rrf_k=60,
                top_n=10,
            )

            retrieval_latency = time.perf_counter() - start

            stage_results = {
                "dense": result["dense_results"],
                "bm25": result["bm25_results"],
                "rrf": result["rrf_results"],
                "cohere": result["reranked_results"],
            }

            # Cohere must not be treated as successful if it failed.
            if result["rerank_status"] != "success":
                stage_results["cohere"] = None

            # Judge the unique profiles from the complete RRF candidate pool.
            profiles = result["scheme_profiles"]

            candidates = [
                {
                    "scheme_id": str(item["scheme_id"]),
                    "text": item["text"],
                }
                for item in profiles
            ]

            judgments = judge_candidates(query, candidates)

            judgment_lookup = {
                item["scheme_id"]: item
                for item in judgments
            }

            query_report = {
                "test_id": query_id,
                "question": query,
                "retrieval_latency": retrieval_latency,
                "final_method": result["final_method"],
                "rerank_status": result["rerank_status"],
                "judged_candidates": len(judgments),
            }

            stage_details = {}

            for stage, schemes in stage_results.items():
                if schemes is None:
                    query_report[f"{stage}_status"] = "failed"
                    continue

                retrieved_ids = extract_ids(schemes)

                id_errors = calculate_id_errors(
                    retrieved_ids,
                    relevant_ids,
                )

                id_metrics = calculate_ir_metrics(
                    query_id,
                    retrieved_ids,
                    relevant_ids,
                )

                semantic_metrics = calculate_semantic_metrics(
                    query_id,
                    retrieved_ids,
                    judgment_lookup,
                )

                stage_details[stage] = {
                    **id_errors,
                    **id_metrics,
                    **semantic_metrics,
                }

                query_report[f"{stage}_retrieved"] = len(retrieved_ids)

                for key, value in id_errors.items():
                    if key not in (
                        "false_positive_ids",
                        "false_negative_ids",
                    ):
                        query_report[f"{stage}_{key}"] = value

                for key, value in id_metrics.items():
                    query_report[f"{stage}_{key}"] = value

                for key, value in semantic_metrics.items():
                    if key != "semantic_ir_metrics":
                        query_report[f"{stage}_{key}"] = value

                for key, value in semantic_metrics[
                    "semantic_ir_metrics"
                ].items():
                    query_report[f"{stage}_judge_{key}"] = value

                query_report[f"{stage}_false_positive_ids"] = json.dumps(
                    id_errors["false_positive_ids"]
                )

                query_report[f"{stage}_false_negative_ids"] = json.dumps(
                    id_errors["false_negative_ids"]
                )

                query_report[f"{stage}_semantic_fp_ids"] = json.dumps(
                    semantic_metrics["semantic_fp_ids"]
                )

                log_langfuse_scores(
                    langfuse,
                    query_id,
                    stage,
                    {
                        **{
                            key: value
                            for key, value in id_errors.items()
                            if isinstance(value, (int, float))
                        },
                        **id_metrics,
                        "semantic_precision": semantic_metrics[
                            "semantic_precision"
                        ],
                        "semantic_fp": semantic_metrics["semantic_fp"],
                    },
                )

            # Store judgments and retrieval output in the trace.
            if trace:
                trace.update(output={
                    "stage_metrics": stage_details,
                    "judgments": judgments,
                    "final_method": result["final_method"],
                    "retrieval_latency": retrieval_latency,
                })

            query_report["judgments"] = json.dumps(
                judgments,
                ensure_ascii=False,
            )

            return query_report

    except Exception:
        logger.exception(
            "Query evaluation failed | test_id=%s",
            query_id,
        )
        raise


def run_evaluation(dataset_path, output_dir, limit=None):
    try:
        load_dotenv()

        run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        rows = load_dataset(dataset_path)

        if limit:
            rows = rows[:limit]

        api_key = os.getenv("COHERE_API_KEY")

        if not api_key:
            raise ValueError("COHERE_API_KEY is missing.")

        cohere_client = cohere.ClientV2(api_key=api_key)
        langfuse = get_langfuse_client()

        discovery_db, _ = load_vectorstores()
        bm25_retriever = build_bm25_retriever(discovery_db)

        logger.info(
            "Evaluation started | run_id=%s | questions=%d",
            run_id,
            len(rows),
        )

        results = []
        failures = []

        for index, row in enumerate(rows, start=1):
            logger.info(
                "Evaluating %d/%d | test_id=%s",
                index,
                len(rows),
                row["test_id"],
            )

            try:
                report = evaluate_query(
                    row=row,
                    discovery_db=discovery_db,
                    bm25_retriever=bm25_retriever,
                    cohere_client=cohere_client,
                    langfuse=langfuse,
                    run_id=run_id,
                )

                results.append(report)

            except Exception as exc:
                failures.append({
                    "test_id": row["test_id"],
                    "error": str(exc),
                })

        langfuse.flush()

        if not results:
            raise RuntimeError("No queries were evaluated successfully.")

        output_path = Path(output_dir) / run_id
        output_path.mkdir(parents=True, exist_ok=True)

        results_df = pd.DataFrame(results)

        csv_path = output_path / "per_query.csv"
        json_path = output_path / "summary.json"

        results_df.to_csv(csv_path, index=False)

        # Aggregate numeric metrics without averaging IDs or text fields.
        numeric_columns = results_df.select_dtypes(
            include="number"
        ).columns

        aggregate = {
            column: float(results_df[column].mean())
            for column in numeric_columns
        }

        summary = {
            "run_id": run_id,
            "dataset": dataset_path,
            "total_questions": len(rows),
            "successful_queries": len(results),
            "failed_queries": len(failures),
            "aggregate_metrics": aggregate,
            "failures": failures,
            "reports": {
                "per_query_csv": str(csv_path),
                "summary_json": str(json_path),
            },
        }

        json_path.write_text(
            json.dumps(summary, indent=4, ensure_ascii=False),
            encoding="utf-8",
        )

        logger.info("Evaluation completed | run_id=%s", run_id)
        logger.info("CSV report: %s", csv_path)
        logger.info("Summary: %s", json_path)

        return summary

    except Exception:
        logger.exception("False retrieval evaluation failed.")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="SEVA_Ai false retrieval evaluation"
    )

    parser.add_argument("--dataset", default=DATASET_PATH)
    parser.add_argument("--output", default=OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=None)

    args = parser.parse_args()

    summary = run_evaluation(
        dataset_path=args.dataset,
        output_dir=args.output,
        limit=args.limit,
    )

    print("\n========== FALSE RETRIEVAL EVALUATION ==========")
    print("Successful:", summary["successful_queries"])
    print("Failed:", summary["failed_queries"])

    for name, value in summary["aggregate_metrics"].items():
        print(f"{name}: {value:.4f}")

    print("\nRun ID:", summary["run_id"])


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    main()
