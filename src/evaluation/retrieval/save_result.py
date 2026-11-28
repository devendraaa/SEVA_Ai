import csv
import json
from datetime import datetime
from pathlib import Path

def save_analysis(captured, judgments):
    try:
        output_dir = Path(r"G:\SEVA_Ai\SEVA_Ai\artifacts\false_retrieval")
        output_dir.mkdir(parents=True, exist_ok=True)

        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        judgment_lookup = {
            item["scheme_id"]: item for item in judgments
        }

        candidate_rows = []

        for candidate in captured["candidates"]:
            judgment = judgment_lookup.get(candidate["scheme_id"], {})

            candidate_rows.append({
                "scheme_id": candidate["scheme_id"],
                "text": candidate["text"],
                "stage_ranks": candidate["stage_ranks"],
                "verdict": judgment.get("verdict", "unjudged"),
                "score": judgment.get("relevance_score"),
                "reason": judgment.get("reason", ""),
            })

        # Compare relevance at each retrieval stage.
        stage_summary = []

        for stage in ["dense", "bm25", "rrf", "cohere"]:
            ranked = [
                row for row in candidate_rows
                if stage in row["stage_ranks"]
            ]

            ranked.sort(
                key=lambda row: row["stage_ranks"][stage]
            )

            for k in [10, 30]:
                top_results = ranked[:k]

                stage_summary.append({
                    "stage": stage,
                    "cutoff": k,
                    "retrieved": len(top_results),
                    "relevant": sum(
                        r["verdict"] == "relevant" for r in top_results
                    ),
                    "partially_relevant": sum(
                        r["verdict"] == "partially_relevant"
                        for r in top_results
                    ),
                    "irrelevant": sum(
                        r["verdict"] == "irrelevant" for r in top_results
                    ),
                    "unjudged": sum(
                        r["verdict"] == "unjudged" for r in top_results
                    ),
                })

        # Complete JSON experiment report.
        report = {
            "run_id": run_id,
            "query": captured["query"],
            "final_method": captured["final_method"],
            "rerank_status": captured["rerank_status"],
            "retrieval_stages": captured["stages"],
            "candidates": candidate_rows,
            "stage_summary": stage_summary,
        }

        json_path = output_dir / f"run_{run_id}.json"
        json_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

        # Candidate-level CSV.
        candidate_path = output_dir / f"candidates_{run_id}.csv"

        with candidate_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "scheme_id", "text", "stage_ranks",
                    "verdict", "score", "reason",
                ],
            )
            writer.writeheader()

            for row in candidate_rows:
                row["stage_ranks"] = json.dumps(row["stage_ranks"])
                writer.writerow(row)

        # Stage-level summary CSV.
        summary_path = output_dir / f"stage_summary_{run_id}.csv"

        with summary_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=stage_summary[0].keys(),
            )
            writer.writeheader()
            writer.writerows(stage_summary)

        print(f"\nResults saved in: {output_dir}")
        print(f"JSON: {json_path}")
        print(f"Candidates: {candidate_path}")
        print(f"Stage summary: {summary_path}")

    except Exception:
        logger.exception("Failed to save retrieval analysis")
        raise
