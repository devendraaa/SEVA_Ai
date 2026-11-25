import logging
import os

from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from observability.tracing import trace_span

load_dotenv()

logger = logging.getLogger(__name__)


def _update_span(span, output):
    if span:
        try:
            span.update(output=output)
        except Exception:
            logger.exception("Failed to update Langfuse span.")


def invoke_llm(prompt: str) -> AIMessage:
    """Invoke Groq, then Gemini, then OpenAI if enabled."""

    providers = [
        {
            "name": "groq",
            "model": os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
            "api_key": "GROQ_API_KEY",
            "client": ChatGroq,
        },
        {
            "name": "gemini",
            "model": os.getenv("GEMINI_MODEL", "gemini-3.6-flash"),
            "api_key": "GEMINI_API_KEY",
            "client": ChatGoogleGenerativeAI,
        },
    ]

    if os.getenv("OPENAI_ENABLED", "false").lower() == "true":
        providers.append({
            "name": "openai",
            "model": os.getenv("OPENAI_MODEL", "gpt-5-mini"),
            "api_key": "OPENAI_API_KEY",
            "client": ChatOpenAI,
        })

    errors = []

    for provider in providers:
        name = provider["name"]
        model_name = provider["model"]

        with trace_span(
            name=f"llm-attempt-{name}",
            input_data={"prompt_logged": False},
            metadata={
                "provider": name,
                "model": model_name,
                "operation": "llm_generation",
            },
        ) as span:

            try:
                api_key = os.getenv(provider["api_key"])

                if not api_key:
                    raise EnvironmentError(
                        f"{provider['api_key']} is not configured."
                    )

                llm = provider["client"](
                    model=model_name,
                    temperature=0,
                )

                response = llm.invoke(prompt)

                usage = response.usage_metadata or {}

                _update_span(span, {
                    "status": "success",
                    "provider": name,
                    "model": model_name,
                    "token_usage": usage,
                })

                logger.info(
                    "LLM generation successful: provider=%s model=%s",
                    name,
                    model_name,
                )

                return response

            except Exception as exc:
                error = f"{name}: {type(exc).__name__}: {exc}"
                errors.append(error)

                _update_span(span, {
                    "status": "failed",
                    "provider": name,
                    "model": model_name,
                    "error": f"{type(exc).__name__}: {exc}",
                })

                logger.warning("LLM provider failed: %s", error)

    raise RuntimeError(
        "All configured LLM providers failed: " + " | ".join(errors)
    )