import logging
from typing import Literal

from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END

from agents.state.agent_state import AgentState
from agents.tools.scheme_tool import search_schemes
from generation.llm.llm import invoke_llm

logger = logging.getLogger(__name__)

from pathlib import Path

PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompts"


def load_prompt(filename: str) -> str:
    try:
        return (PROMPT_DIR / filename).read_text(encoding="utf-8")
    except Exception:
        logger.exception("Failed to load prompt: %s", filename)
        raise


class UserInput(BaseModel):
    query: str = Field(min_length=3, max_length=1000)


class IntentResult(BaseModel):
    intent: Literal[
        "scheme_discovery",
        "scheme_details",
        "eligibility_check",
        "scheme_comparison",
        "hospital_search",
        "general",
    ]


class QueryEntities(BaseModel):
    age: int | None = None
    occupation: str | None = None
    location: str | None = None
    scheme_name: str | None = None


class QueryUnderstanding(BaseModel):
    rewritten_query: str
    entities: QueryEntities
    missing_information: list[str] = Field(default_factory=list)


def classify_intent(state: AgentState) -> dict:
    try:
        prompt = load_prompt("intent_classification_v1.txt")
        prompt = prompt.replace("{query}", state["query"])
        response = invoke_llm(prompt)
        result = IntentResult.model_validate_json(response.content)

        logger.info("Intent classified | intent=%s", result.intent)
        return {"intent": result.intent}

    except Exception:
        logger.exception("Intent classification failed.")
        raise


def understand_query(state: AgentState) -> dict:
    try:
        prompt = load_prompt("query_understanding_v1.txt")
        prompt = prompt.replace("{intent}", state["intent"])
        prompt = prompt.replace("{query}", state["query"])

        response = invoke_llm(prompt)
        result = QueryUnderstanding.model_validate_json(response.content)

        logger.info("Query understanding completed.")

        return {
            "rewritten_query": result.rewritten_query,
            "entities": result.entities.model_dump(),
            "missing_information": result.missing_information,
        }

    except Exception:
        logger.exception("Query understanding failed.")
        raise


def route_intent(state: AgentState) -> str:
    try:
        scheme_intents = {
            "scheme_discovery",
            "scheme_details",
            "eligibility_check",
            "scheme_comparison",
        }

        return (
            "scheme_search"
            if state["intent"] in scheme_intents
            else "general_help"
        )

    except Exception:
        logger.exception("Intent routing failed.")
        raise


def scheme_search_node(state: AgentState) -> dict:
    try:
        query = state.get("rewritten_query", state["query"])
        result = search_schemes.invoke({"query": query})

        return {"tool_result": result, "answer": result}

    except Exception:
        logger.exception("Scheme search failed.")
        raise


def general_help_node(state: AgentState) -> dict:
    try:
        if state["intent"] == "hospital_search":
            answer = "Hospital search workflow is not connected yet."
        else:
            answer = "This query is outside the currently connected scheme-search workflow."

        return {"answer": answer}

    except Exception:
        logger.exception("General response failed.")
        raise


def build_agent():
    try:
        graph = StateGraph(AgentState)

        graph.add_node("classify_intent", classify_intent)
        graph.add_node("understand_query", understand_query)
        graph.add_node("scheme_search", scheme_search_node)
        graph.add_node("general_help", general_help_node)

        graph.add_edge(START, "classify_intent")
        graph.add_edge("classify_intent", "understand_query")

        graph.add_conditional_edges(
            "understand_query",
            route_intent,
            {
                "scheme_search": "scheme_search",
                "general_help": "general_help",
            },
        )

        graph.add_edge("scheme_search", END)
        graph.add_edge("general_help", END)

        logger.info("SEVA_Ai agent graph compiled.")
        return graph.compile()

    except Exception:
        logger.exception("Agent graph creation failed.")
        raise


def run_agent(query: str) -> dict:
    try:
        validated = UserInput(query=query)
        result = build_agent().invoke({"query": validated.query})

        logger.info("Agent completed | intent=%s", result.get("intent"))
        return result

    except Exception:
        logger.exception("Agent execution failed.")
        raise