from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="User's government scheme related question",
    )

class SchemeSource(BaseModel):
    scheme_id: str
    rank: int

class ChatResponse(BaseModel):
    query: str
    answer: str
    sources: list[SchemeSource]
    retrieval_method: str
    retrieved_count: int
