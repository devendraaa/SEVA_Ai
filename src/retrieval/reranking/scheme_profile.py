import logging
import time

logger = logging.getLogger(__name__)

FIELD_PRIORITY = {
    "brief_description": 1,
    "details": 2,
    "benefits": 3,
    "eligibility": 4,
    "exclusions": 5,
    "application_process": 6,
    "documents_required": 7,
}


def build_scheme_profiles(
    discovery_db,
    fused_results: list[dict],
    max_chars: int = 3500,
) -> list[dict]:
    """Build text profiles for fused scheme candidates."""

    start_time = time.perf_counter()

    try:
        scheme_ids = [
            item["scheme_id"]
            for item in fused_results
        ]

        if not scheme_ids:
            logger.info("Scheme profile generation skipped | no candidates")
            return []

        data = discovery_db.get(
            where={"scheme_id": {"$in": scheme_ids}},
            include=["documents", "metadatas"],
        )

        grouped_documents = {}

        for content, metadata in zip(
            data["documents"],
            data["metadatas"],
        ):
            if not content or not metadata:
                continue

            scheme_id = metadata.get("scheme_id")
            source_field = metadata.get("source_field", "")

            if scheme_id not in scheme_ids:
                continue

            grouped_documents.setdefault(
                scheme_id,
                [],
            ).append({
                "content": content,
                "priority": FIELD_PRIORITY.get(source_field, 99),
            })

        profiles = []

        for candidate in fused_results:
            scheme_id = candidate["scheme_id"]

            documents = grouped_documents.get(scheme_id, [])

            documents.sort(
                key=lambda item: item["priority"]
            )

            profile_text = "\n\n".join(
                item["content"]
                for item in documents
            )

            profile_text = profile_text[:max_chars]

            if not profile_text:
                continue

            profiles.append({
                "scheme_id": scheme_id,
                "text": profile_text,
                "rrf_score": candidate["rrf_score"],
                "rank": candidate["rank"],
            })

        latency = time.perf_counter() - start_time

        logger.info(
            "Scheme profiles generated | candidates=%d | profiles=%d | latency=%.4fs",
            len(fused_results),
            len(profiles),
            latency,
        )

        return profiles

    except Exception:
        logger.exception("Scheme profile generation failed")
        raise
