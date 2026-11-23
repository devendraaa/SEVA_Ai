from contextlib import asynccontextmanager
from fastapi import FastAPI
from api.routes import health
from core.logging import setup_logging
from indexing import load_vectorstores

logger = setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("SEVA_AI application starting")

    # Load existing embedding model and vector stores
    discovery_db, detail_db = load_vectorstores()

    app.state.discovery_db = discovery_db
    app.state.detail_db = detail_db

    logger.info("Chroma vector stores initialized successfully")

    yield

    # Application shutdown
    app.state.discovery_db = None
    app.state.detail_db = None

    logger.info("SEVA_AI application shutting down")


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
        "status": "running"
    }