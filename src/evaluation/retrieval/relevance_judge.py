
import json
import logging
import re
import time

from generation.llm.llm import invoke_llm

logger = logging.getLogger(__name__)

BATCH_SIZE = 5


def _response_text(response):
    content = response.content

    if isinstance(content, list):
        return " ".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
        )

    return str(content)


def _parse_json(text):
    text = re.sub(r"```(?:json)?|```", "", text).strip()

    start = text.find("[")
    end = text.rfind("]")

    if start == -1 or end == -1:
        raise ValueError("Judge did not return a JSON array.")

    return json.loads(text[start:end + 1])


def judge_candidates(query: str, candidates: list[dict]) -> list[dict]:
    """Judge scheme relevance using the existing LLM fallback."""

    judgments = []

    try:
        for start in range(0, len(candidates), BATCH_SIZE):
            batch = candidates[start:start + BATCH_SIZE]
            batch_start = time.perf_counter()

            prompt = f"""
                    You are evaluating an Indian government scheme retrieval system.

                    User question:
                    {query}

                    Evaluate each scheme independently.

                    Rules:
                    - Judge semantic relevance to the question.
                    - Schemes from ANY Indian state, union territory, or central government
                    can be relevant.
                    - Do not reject a scheme because of its state or location.
                    - Do not assume eligibility when the scheme text does not support it.
                    - Use only the supplied scheme information.
                    - "partially_relevant" means related but only partially addresses the request.
                    - "irrelevant" means it does not meaningfully address the request.

                    Return ONLY a JSON array:
                    [
                    {{
                        "scheme_id": "exact_id",
                        "verdict": "relevant | partially_relevant | irrelevant",
                        "relevance_score": 0.0,
                        "reason": "short explanation"
                    }}
                    ]

                        Schemes:
                        {json.dumps(batch, ensure_ascii=False)}
                        """

            response = invoke_llm(prompt)
            parsed = _parse_json(_response_text(response))

            allowed_ids = {str(item["scheme_id"]) for item in batch}

            for item in parsed:
                scheme_id = str(item.get("scheme_id", ""))

                if scheme_id not in allowed_ids:
                    continue

                verdict = item.get("verdict", "").lower()

                if verdict not in {
                    "relevant",
                    "partially_relevant",
                    "irrelevant",
                }:
                    continue

                score = max(
                    0.0,
                    min(1.0, float(item.get("relevance_score", 0))),
                )

                judgments.append({
                    "scheme_id": scheme_id,
                    "verdict": verdict,
                    "relevance_score": score,
                    "reason": str(item.get("reason", "")),
                })

            logger.info(
                "Judge batch completed | batch=%d | judged=%d | latency=%.2fs",
                start // BATCH_SIZE + 1,
                len(parsed),
                time.perf_counter() - batch_start,
            )

        logger.info(
            "Semantic judging completed | candidates=%d | judged=%d",
            len(candidates),
            len(judgments),
        )

        return judgments

    except Exception:
        logger.exception("Semantic relevance judging failed.")
        raise
