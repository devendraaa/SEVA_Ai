
import argparse
import json
import logging
import os

from datetime import datetime
from pathlib import Path

import cohere
import pandas as pd

from dotenv import load_dotenv

from indexing import load_vectorstores

from retrieval.bm25.retrieval import build_bm25_retriever
from retrieval.hybrid.retrieval import retrieve_hybrid_schemes

from evaluation.datasets.golden_dataset import load_golden_dataset
from evaluation.metrics.retrieval_metrics import calculate_retrieval_metrics

from observability.langfuse_client import get_langfuse_client
from observability.tracing import trace_span


logger = logging.getLogger(__name__)

K_VALUES = (1, 3, 5, 10)
STAGES = ("dense", "bm25", "rrf", "cohere")

DATASET_PATH = (
    "notebooks/golden_dataset/discovery/"
    "discovery_131_validation_input.jsonl"
)


def extract_scheme_ids(retrieved_schemes: list) -> list:
    """Extract unique scheme IDs while preserving ranking."""

    try:
        return list(dict.fromkeys(
            scheme["scheme_id"]
            for scheme in retrieved_schemes
            if scheme.get("scheme_id")
        ))

    except Exception:
        logger.exception("Failed to extract scheme IDs.")
        raise


def evaluate_retrieval_stages(
    result: dict,
    relevant_ids: list,
) -> tuple[dict, dict]:
    """Calculate metrics independently for each retrieval stage."""

    try:
        stage_results = {
            "dense": result["dense_results"],
            "bm25": result["bm25_results"],
            "rrf": result["rrf_results"],
            "cohere": result["reranked_results"],
        }

        stage_metrics = {}
        stage_report = {}

        for stage, schemes in stage_results.items():

            # Cohere failure must not be evaluated as RRF fallback.
            if stage == "cohere" and result["rerank_status"] != "success":
                stage_metrics[stage] = None

                for key in (
                    "mrr",
                    "map",
                    "candidate_recall",
                    *[
                        metric
                        for k in K_VALUES
                        for metric in (f"recall@{k}", f"ndcg@{k}")
                    ],
                ):
                    stage_report[f"{stage}_{key}"] = None

                continue

            retrieved_ids = extract_scheme_ids(schemes)

            metrics = calculate_retrieval_metrics(
                retrieved_ids=retrieved_ids,
                relevant_ids=relevant_ids,
                k_values=K_VALUES,
            )

            relevant_set = set(relevant_ids)
            matched = set(retrieved_ids) & relevant_set

            metrics["candidate_recall"] = (
                len(matched) / len(relevant_set)
                if relevant_set else 0.0
            )

            metrics["relevant_rank"] = (
                min(
                    retrieved_ids.index(scheme_id) + 1
                    for scheme_id in matched
                )
                if matched else None
            )

            stage_metrics[stage] = metrics

            stage_report.update({
                f"{stage}_{key}": value
                for key, value in metrics.items()
            })

        return stage_metrics, stage_report

    except Exception:
        logger.exception("Retrieval stage evaluation failed.")
        raise


def log_stage_scores(
    langfuse,
    stage_metrics: dict,
    test_id: str,
    run_id: str,
) -> None:
    """Record retrieval metrics against the active Langfuse trace."""

    try:
        for stage, metrics in stage_metrics.items():

            if metrics is None:
                continue

            for metric_name, value in metrics.items():

                if metric_name == "relevant_rank" or value is None:
                    continue

                langfuse.score_current_trace(
                    name=f"{stage}_{metric_name}",
                    value=float(value),
                    data_type="NUMERIC",
                    metadata={
                        "test_id": str(test_id),
                        "run_id": run_id,
                        "retrieval_stage": stage,
                    },
                )

    except Exception:
        logger.exception(
            "Failed to record Langfuse scores | test_id=%s",
            test_id,
        )


def run_hybrid_evaluation(
    dataset_path: str,
    output_dir: str,
    limit: int | None = None,
) -> dict:

    try:
        load_dotenv()

        run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        # 1. Dataset
        golden_df = load_golden_dataset(dataset_path)

        if limit is not None:
            if limit <= 0:
                raise ValueError("Evaluation limit must be positive.")

            golden_df = golden_df.head(limit)

        logger.info("Evaluation questions: %d", len(golden_df))

        # 2. Initialize clients
        api_key = os.getenv("COHERE_API_KEY")

        if not api_key:
            raise ValueError("COHERE_API_KEY is missing.")

        cohere_client = cohere.ClientV2(api_key=api_key)
        langfuse = get_langfuse_client()

        # 3. Load retrieval resources
        discovery_db, _ = load_vectorstores()
        bm25_retriever = build_bm25_retriever(discovery_db)

        evaluation_results = []
        failed_queries = []

        # 4. Evaluate queries
        for index, row in enumerate(
            golden_df.itertuples(index=False),
            start=1,
        ):

            try:
                query = row.question
                relevant_ids = row.relevant_scheme_ids

                if not relevant_ids:
                    raise ValueError(
                        f"No relevant scheme IDs: {row.test_id}"
                    )

                logger.info(
                    "Evaluating %d/%d | test_id=%s",
                    index,
                    len(golden_df),
                    row.test_id,
                )

                with trace_span(
                    name="hybrid-retrieval-evaluation",
                    input_data={
                        "test_id": str(row.test_id),
                        "question": query,
                        "relevant_scheme_ids": relevant_ids,
                    },
                    metadata={
                        "run_id": run_id,
                        "dataset": "golden_test",
                        "retrieval_method": "Dense + BM25 + RRF + Cohere",
                    },
                ) as trace:

                    # Retrieval pipeline creates its own child observations.
                    result = retrieve_hybrid_schemes(
                        query=query,
                        discovery_db=discovery_db,
                        bm25_retriever=bm25_retriever,
                        cohere_client=cohere_client,
                        retrieval_k=30,
                        rrf_k=60,
                        top_n=10,
                    )

                    stage_metrics, stage_report = (
                        evaluate_retrieval_stages(
                            result=result,
                            relevant_ids=relevant_ids,
                        )
                    )

                    # Final metrics follow the actual final retrieval method.
                    final_stage = result["final_method"]
                    final_stage = (
                        "rrf" if final_stage == "rrf_fallback"
                        else final_stage
                    )

                    metrics = {
                        key: value
                        for key, value in stage_metrics[final_stage].items()
                        if key not in ("candidate_recall", "relevant_rank")
                    }

                    retrieved_ids = extract_scheme_ids(
                        result["retrieved_schemes"]
                    )

                    # Record evaluation scores.
                    log_stage_scores(
                        langfuse=langfuse,
                        stage_metrics=stage_metrics,
                        test_id=str(row.test_id),
                        run_id=run_id,
                    )

                    if trace:
                        trace.update(
                            output={
                                "retrieved_scheme_ids": retrieved_ids,
                                "final_method": result["final_method"],
                                "rerank_status": result["rerank_status"],
                                "stage_metrics": stage_metrics,
                            }
                        )

                query_result = {
                    "test_id": row.test_id,
                    "question": query,
                    "relevant_scheme_ids": relevant_ids,
                    "retrieved_scheme_ids": retrieved_ids,
                    "retrieved_count": len(retrieved_ids),
                    "final_method": result["final_method"],
                    "rerank_status": result["rerank_status"],
                    **metrics,
                    **stage_report,
                }

                # Keep existing latency report.
                for stage, latency in result["latency"].items():
                    query_result[f"latency_{stage}"] = latency

                evaluation_results.append(query_result)

            except Exception as exc:
                logger.exception(
                    "Query evaluation failed | test_id=%s",
                    row.test_id,
                )

                failed_queries.append({
                    "test_id": row.test_id,
                    "error": str(exc),
                })

        # 5. Flush Langfuse
        try:
            langfuse.flush()
            logger.info("Langfuse observations flushed.")

        except Exception:
            logger.exception("Langfuse flush failed.")

        if not evaluation_results:
            raise RuntimeError("No queries evaluated successfully.")

        # 6. Aggregate metrics
        results_df = pd.DataFrame(evaluation_results)

        metric_columns = [
            column
            for column in results_df.columns
            if column in (
                "mrr",
                "map",
                *[
                    metric
                    for k in K_VALUES
                    for metric in (f"recall@{k}", f"ndcg@{k}")
                ],
            )
            or any(
                column == f"{stage}_{metric}"
                for stage in STAGES
                for metric in (
                    "mrr",
                    "map",
                    "candidate_recall",
                    *[
                        metric
                        for k in K_VALUES
                        for metric in (f"recall@{k}", f"ndcg@{k}")
                    ],
                )
            )
        ]

        aggregate_metrics = {
            metric: float(results_df[metric].mean())
            for metric in metric_columns
        }

        # 7. Aggregate latency
        latency_columns = [
            column
            for column in results_df.columns
            if column.startswith("latency_")
        ]

        average_latency = {
            column.replace("latency_", ""):
                float(results_df[column].mean())
            for column in latency_columns
        }

        # 8. Save reports
        output_path = Path(output_dir) / run_id
        output_path.mkdir(parents=True, exist_ok=True)

        csv_path = output_path / "hybrid_per_query.csv"
        json_path = output_path / "hybrid_summary.json"

        results_df.to_csv(csv_path, index=False)

        summary = {
            "run_id": run_id,
            "retrieval_method": "Dense + BM25 + RRF + Cohere",
            "total_questions": len(golden_df),
            "successful_queries": len(evaluation_results),
            "failed_queries": len(failed_queries),
            "average_latency": average_latency,
            "metrics": aggregate_metrics,
            "failures": failed_queries,
        }

        with json_path.open("w", encoding="utf-8") as file:
            json.dump(summary, file, indent=4, ensure_ascii=False)

        logger.info("Evaluation completed | run_id=%s", run_id)
        logger.info("CSV: %s", csv_path)
        logger.info("JSON: %s", json_path)

        return summary

    except Exception:
        logger.exception("Hybrid evaluation failed.")
        raise


def main():

    parser = argparse.ArgumentParser(
        description="SEVA_Ai Hybrid Retrieval Evaluation"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of golden questions to evaluate.",
    )

    args = parser.parse_args()

    try:
        summary = run_hybrid_evaluation(
            dataset_path=DATASET_PATH,
            output_dir="artifacts/evaluation/hybrid",
            limit=args.limit,
        )

        print("\n========== HYBRID EVALUATION ==========")
        print(f"Successful: {summary['successful_queries']}")
        print(f"Failed: {summary['failed_queries']}")

        print("\nMetrics:")
        for metric, value in summary["metrics"].items():
            print(f"{metric}: {value:.4f}")

        print("\nAverage latency:")
        for stage, latency in summary["average_latency"].items():
            print(f"{stage}: {latency:.4f}s")

        print("\nRun ID:", summary["run_id"])

    except Exception:
        logger.exception("Evaluation script failed.")
        raise


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | %(levelname)s | "
            "%(name)s | %(message)s"
        ),
    )

    main()
