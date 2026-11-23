
import logging
from indexing import load_vectorstores
from retrieval.details_ret.metadata_filter import retrieve_scheme_details

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

def main():
    try:
        # Load existing vector stores
        _, detail_db = load_vectorstores()

        scheme_id = "kxiR9pwB5uF80_MGNmL_"

        # Retrieve all children for this scheme
        result = retrieve_scheme_details(
            scheme_id=scheme_id,
            detail_db=detail_db,
        )

        documents = result["documents"]

        print("\nScheme ID:", result["scheme_id"])
        print("Retrieved chunks:", result["retrieved_chunks"])
        print("Latency:", round(result["latency"], 4), "seconds")

        # Inspect metadata fields
        if documents:
            print("\nMetadata keys:")
            print(list(documents[0]["metadata"].keys()))

        # Count chunks by section
        section_counts = {}

        for document in documents:
            metadata = document["metadata"]

            section = (
                metadata.get("source_field")
                or metadata.get("child_type")
                or metadata.get("section")
                or "unknown"
            )

            section_counts[section] = (
                section_counts.get(section, 0) + 1
            )

        print("\nChunks by section:")

        for section, count in sorted(section_counts.items()):
            print(f"{section}: {count}")

        # Preview first three chunks
        print("\nFirst 3 retrieved chunks:")

        for index, document in enumerate(documents[:3], start=1):
            print(f"\n--- Chunk {index} ---")
            print("Metadata:", document["metadata"])
            print("Content:", document["page_content"][:500])

    except Exception:
        logging.exception("Detail metadata retrieval test failed")
        raise


if __name__ == "__main__":
    main()
