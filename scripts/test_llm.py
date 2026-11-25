import logging

from generation.llm.llm import invoke_llm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

try:
    response = invoke_llm(
        "Explain government health insurance schemes in India "
        "in three simple sentences."
    )

    print("\nGenerated response:\n")
    print(response.content)

    print("\nToken usage:")
    print(response.usage_metadata)

except Exception:
    logging.exception("LLM test failed.")