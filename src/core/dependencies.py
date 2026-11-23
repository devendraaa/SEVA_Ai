
from fastapi import Request

from langchain_chroma import Chroma


def get_discovery_db(request: Request) -> Chroma:
    """Return the initialized Discovery vector store."""
    return request.app.state.discovery_db


def get_detail_db(request: Request) -> Chroma:
    """Return the initialized Detail vector store."""
    return request.app.state.detail_db
