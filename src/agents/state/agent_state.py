from typing import TypedDict


class AgentState(TypedDict, total=False):
    query: str
    intent: str
    rewritten_query: str
    entities: dict
    missing_information: list[str]
    tool_result: str
    answer: str