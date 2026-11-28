from typing import TypedDict


class AgentState(TypedDict, total=False):
    query: str
    intent: str
    tool_result: str
    answer: str