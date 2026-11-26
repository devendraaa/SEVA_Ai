
import logging
import os
import re
from pathlib import Path

logger = logging.getLogger(__name__)

PROMPT_DIR = Path(__file__).resolve().parent / "versions" / "chat"
DEFAULT_VERSION = "v1"

def build_chat_prompt(query: str, context: str, version: str = None):
    try:
        version = version or os.getenv(
            "CHAT_PROMPT_VERSION", DEFAULT_VERSION
        )

        if not re.fullmatch(r"v[0-9]+", version):
            raise ValueError(f"Invalid prompt version: {version}")

        prompt_path = PROMPT_DIR / f"{version}.txt"

        if not prompt_path.is_file():
            raise FileNotFoundError(
                f"Prompt file not found: {prompt_path}"
            )

        template = prompt_path.read_text(encoding="utf-8")

        prompt = (
            template
            .replace("{{QUERY}}", query)
            .replace("{{CONTEXT}}", context)
        )

        if "{{QUERY}}" in prompt or "{{CONTEXT}}" in prompt:
            raise ValueError("Prompt contains unresolved placeholders.")

        logger.info("Chat prompt loaded | version=%s", version)

        return prompt, version

    except Exception:
        logger.exception("Failed to build chat prompt")
        raise
