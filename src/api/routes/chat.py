import logging
from generation.prompts.prompt_loader import build_chat_prompt

from fastapi import APIRouter, Depends, HTTPException

from api.schemas.chat import ChatRequest, ChatResponse
from core.dependencies import (
    get_discovery_db,
    get_bm25_retriever,
    get_cohere_client,
)

from retrieval.hybrid.retrieval import retrieve_hybrid_schemes
from generation.llm.llm import invoke_llm

from observability.tracing import trace_span
from observability.langfuse_client import get_langfuse_client


logger = logging.getLogger(__name__)

router = APIRouter(tags=["Chat"])

RETRIEVAL_K = 30
TOP_N = 5
RRF_K = 60
MAX_CONTEXT_CHARS = 18000


def prepare_context(retrieval_result: dict) -> tuple[str, list]:
    """Prepare context using final retrieved scheme profiles."""

    try:
        profiles = retrieval_result["scheme_profiles"]

        profile_lookup = {
            profile["scheme_id"]: profile
            for profile in profiles
        }

        context_parts = []
        sources = []
        total_chars = 0

        for scheme in retrieval_result["retrieved_schemes"][:TOP_N]:
            scheme_id = scheme["scheme_id"]
            profile = profile_lookup.get(scheme_id)

            if not profile:
                continue

            text = profile["text"]
            remaining_chars = MAX_CONTEXT_CHARS - total_chars

            if remaining_chars <= 0:
                break

            text = text[:remaining_chars]

            context_parts.append(
                f"SCHEME ID: {scheme_id}\n"
                f"SCHEME INFORMATION:\n{text}"
            )

            sources.append({
                "scheme_id": scheme_id,
                "rank": scheme.get("rank", len(sources) + 1),
            })

            total_chars += len(text)

        context = "\n\n".join(context_parts)

        if not context:
            raise ValueError("No usable scheme context retrieved.")

        logger.info(
            "Context prepared | schemes=%d | characters=%d",
            len(sources),
            len(context),
        )

        return context, sources

    except Exception:
        logger.exception("Context preparation failed")
        raise


def extract_answer(response) -> str:
    """Normalize LangChain AIMessage content."""

    content = response.content

    if isinstance(content, list):
        content = "\n".join(
            str(block.get("text", block))
            if isinstance(block, dict)
            else str(block)
            for block in content
        )

    answer = str(content).strip()

    if not answer:
        raise ValueError("LLM returned an empty answer.")

    return answer

@router.post("", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    discovery_db=Depends(get_discovery_db),
    bm25_retriever=Depends(get_bm25_retriever),
    cohere_client=Depends(get_cohere_client),
):
    """Execute the SEVA_Ai hybrid RAG pipeline."""

    query = request.query.strip()

    try:
        logger.info("Chat request received | query=%s", query)

        retrieval_result = retrieve_hybrid_schemes(
            query=query,
            discovery_db=discovery_db,
            bm25_retriever=bm25_retriever,
            cohere_client=cohere_client,
            retrieval_k=RETRIEVAL_K,
            rrf_k=RRF_K,
            top_n=TOP_N,
        )

        context, sources = prepare_context(retrieval_result)

        # Build prompt BEFORE creating the trace
        prompt, prompt_version = build_chat_prompt(
            query=query,
            context=context,
        )

        with trace_span(
            name="seva-chat-api",
            input_data={"query": query},
            metadata={
                "pipeline": "dense-bm25-rrf-cohere-llm",
                "retrieval_k": RETRIEVAL_K,
                "top_n": TOP_N,
                "prompt_name": "chat",
                "prompt_version": prompt_version,
            },
        ) as span:

            response = invoke_llm(prompt)
            answer = extract_answer(response)

            result = {
                "query": query,
                "answer": answer,
                "sources": sources,
                "retrieval_method": retrieval_result["final_method"],
                "retrieved_count": len(retrieval_result["retrieved_schemes"]),
            }

            if span:
                span.update(output=result)

        return result

    except Exception:
        logger.exception("Chat request failed")

        raise HTTPException(
            status_code=500,
            detail="SEVA_Ai could not process your request. Please try again.",
        )

    finally:
        try:
            get_langfuse_client().flush()
        except Exception:
            logger.exception("Langfuse flush failed")
