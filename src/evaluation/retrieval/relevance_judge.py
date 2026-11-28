import json
import logging
import re

from json_repair import repair_json  # uv add json-repair

from generation.llm.llm import invoke_llm

logger = logging.getLogger(__name__)

BATCH_SIZE = 5
MAX_RETRIES = 3
VALID_VERDICTS = {"relevant", "partially_relevant", "irrelevant"}


def _extract_text(content) -> str:
    if isinstance(content, list):
        return " ".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
        )
    return str(content)


def _parse_json_array(content: str) -> list:
    content = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", content.strip())

    start = content.find("[")
    end = content.rfind("]")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON array found in judge response.")

    candidate = content[start:end + 1]
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        parsed = json.loads(repair_json(candidate))

    if not isinstance(parsed, list):
        raise ValueError("Judge response is not a JSON array.")
    return parsed


def _build_prompt(query, schemes) -> str:
    return f"""
You are evaluating an Indian government scheme retrieval system.

User query: {query}

Evaluate each scheme independently.

Rules:
- Schemes from any Indian state, UT or central government can be relevant.
- Judge semantic relevance to the query.
- Use only the supplied scheme information.
- Do not assume eligibility without evidence.

Classify each scheme as: relevant, partially_relevant, or irrelevant.

Output rules:
- Return ONLY a JSON array, no markdown, no extra text.
- "reason" must be under 15 words and must NOT contain double quotes.
- "relevance_score" must be a number between 0 and 1.

Format:
[
  {{"scheme_id": "exact_id", "verdict": "relevant", "relevance_score": 0.9, "reason": "short reason"}}
]

Schemes:
{json.dumps(schemes, ensure_ascii=False)}
"""


def _judge_batch(query, batch):
    schemes = [{"scheme_id": str(i["scheme_id"]), "text": i["text"]} for i in batch]
    prompt = _build_prompt(query, schemes)

    parsed = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = invoke_llm(prompt)
            raw = _extract_text(response.content)
            parsed = _parse_json_array(raw)
            break
        except Exception as e:
            logger.warning(
                "Judge parse failed (attempt %d/%d): %s | RAW: %s",
                attempt, MAX_RETRIES, e, locals().get("raw", "")[:1500],
            )

    if parsed is None:
        return None

    allowed_ids = {str(i["scheme_id"]) for i in batch}
    results = []

    for item in parsed:
        if not isinstance(item, dict):
            continue

        scheme_id = str(item.get("scheme_id", ""))
        verdict = str(item.get("verdict", "")).strip().lower()

        if scheme_id not in allowed_ids:
            logger.warning("Judge returned unknown scheme_id=%s", scheme_id)
            continue
        if verdict not in VALID_VERDICTS:
            logger.warning("Invalid verdict '%s' for scheme_id=%s", verdict, scheme_id)
            continue

        try:
            score = max(0.0, min(1.0, float(item.get("relevance_score", 0))))
        except (TypeError, ValueError):
            score = {"relevant": 1.0, "partially_relevant": 0.5}.get(verdict, 0.0)

        results.append({
            "scheme_id": scheme_id,
            "verdict": verdict,
            "relevance_score": score,
            "reason": str(item.get("reason", "")),
            "judge_failed": False,
        })

    return results


def judge_candidates(query, candidates):
    judgments = []
    failed_batches = 0

    for start in range(0, len(candidates), BATCH_SIZE):
        batch = candidates[start:start + BATCH_SIZE]
        batch_no = start // BATCH_SIZE + 1

        results = _judge_batch(query, batch)

        if results is None:
            failed_batches += 1
            logger.error("Judge batch %d failed after %d retries", batch_no, MAX_RETRIES)
            results = [
                {
                    "scheme_id": str(i["scheme_id"]),
                    "verdict": None,
                    "relevance_score": None,
                    "reason": "",
                    "judge_failed": True,
                }
                for i in batch
            ]

        judgments.extend(results)
        logger.info("Judge batch %d done | judgments so far=%d", batch_no, len(judgments))

    logger.info(
        "Judging completed | candidates=%d | judged=%d | failed_batches=%d",
        len(candidates), len(judgments), failed_batches,
    )
    return judgments