
from fastapi import Request

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from cohere import ClientV2


def get_embeddings(request: Request) -> HuggingFaceEmbeddings:
    return request.app.state.embeddings


def get_discovery_db(request: Request) -> Chroma:
    return request.app.state.discovery_db


def get_detail_db(request: Request) -> Chroma:
    return request.app.state.detail_db


def get_bm25_retriever(request: Request) -> BM25Retriever:
    return request.app.state.bm25_retriever


def get_cohere_client(request: Request) -> ClientV2:
    return request.app.state.cohere_client
