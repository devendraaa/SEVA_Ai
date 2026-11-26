import logging
from contextlib import asynccontextmanager
from time import perf_counter
from fastapi import FastAPI
from api.routes import health
from core.logging import setup_logging
from indexing import load_embedding_model, load_vectorstores

logger = setup_logging()

@asynccontextmanager
async def lifespan(app: FastAPI):
    startup_time = perf_counter()

    logger.info("Starting SEVA_AI application...")

    try:
        # Load embedding model once
        embeddings = load_embedding_model()

        # Load both Chroma stores using the same model instance
        discovery_db, detail_db = load_vectorstores(embeddings)

        # Store shared resources
        app.state.embeddings = embeddings
        app.state.discovery_db = discovery_db
        app.state.detail_db = detail_db

        logger.info(
            "Application initialized | Discovery: %s | Detail: %s | Time: %.2fs",
            discovery_db._collection.count(),
            detail_db._collection.count(),
            perf_counter() - startup_time,
        )

        yield

    except Exception:
        logger.exception("SEVA_AI startup failed")
        raise

    finally:
        app.state.embeddings = None
        app.state.discovery_db = None
        app.state.detail_db = None

        logger.info("SEVA_AI shutdown completed")


app = FastAPI(
    title="SEVA_AI",
    description="AI-powered Government Scheme Discovery Platform",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(health.router, prefix="/api/v1")


@app.get("/")
async def root():
    return {
        "message": "Welcome to SEVA_AI",
        "status": "running",
    }
