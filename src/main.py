
import logging
import os

import cohere
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from time import perf_counter
from fastapi import FastAPI

from api.routes import health, chat
from core.logging import setup_logging
from indexing import load_embedding_model, load_vectorstores
from retrieval.bm25.retrieval import build_bm25_retriever


load_dotenv()
logger = setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    startup_time = perf_counter()

    logger.info("Starting SEVA_Ai application...")

    try:
        # 1. Load embedding model
        embeddings = load_embedding_model()

        # 2. Load Chroma vector stores
        discovery_db, detail_db = load_vectorstores(embeddings)

        # 3. Build BM25 index once
        bm25_retriever = build_bm25_retriever(discovery_db)

        # 4. Initialize Cohere client once
        cohere_api_key = os.getenv("COHERE_API_KEY")

        if not cohere_api_key:
            raise ValueError("COHERE_API_KEY is missing.")

        cohere_client = cohere.ClientV2(
            api_key=cohere_api_key
        )

        # 5. Register shared application resources
        app.state.embeddings = embeddings
        app.state.discovery_db = discovery_db
        app.state.detail_db = detail_db
        app.state.bm25_retriever = bm25_retriever
        app.state.cohere_client = cohere_client

        logger.info(
            "SEVA_Ai startup completed | "
            "Discovery=%d | Detail=%d | Time=%.2fs",
            discovery_db._collection.count(),
            detail_db._collection.count(),
            perf_counter() - startup_time,
        )

        yield

    except Exception:
        logger.exception("SEVA_Ai application startup failed")
        raise

    finally:
        app.state.embeddings = None
        app.state.discovery_db = None
        app.state.detail_db = None
        app.state.bm25_retriever = None
        app.state.cohere_client = None

        logger.info("SEVA_Ai application shutdown completed")


app = FastAPI(
    title="SEVA_Ai",
    description="AI-powered Government Scheme Discovery Platform",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(health.router, prefix="/api/v1")
app.include_router(chat.router, prefix="/api/v1/chat")


@app.get("/")
async def root():
    return {
        "message": "Welcome to SEVA_Ai",
        "status": "running",
    }
