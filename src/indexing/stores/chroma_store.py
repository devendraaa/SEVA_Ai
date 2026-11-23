
import logging
from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]

VECTORSTORE_DIR = (
    PROJECT_ROOT / "artifacts" / "indexing" / "vectorstores"
)

MODEL_PATH = Path(
    r"G:\SEVA_Ai\SEVA_Ai\models\all-MiniLM-L6-v2"
)

REQUIRED_FILES = [
    "config.json",
    "model.safetensors",
    "modules.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
    "1_Pooling/config.json",
]


def load_embedding_model():
    """Load the existing local embedding model."""

    missing_files = [
        file for file in REQUIRED_FILES
        if not (MODEL_PATH / file).exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            f"Missing embedding model files: {missing_files}"
        )

    logger.info("Loading local embedding model: %s", MODEL_PATH)

    return HuggingFaceEmbeddings(
        model_name=str(MODEL_PATH),
        model_kwargs={"local_files_only": True},
        encode_kwargs={"normalize_embeddings": False},
    )


def load_vectorstores():
    """Load existing discovery and detail Chroma collections."""

    embeddings = load_embedding_model()

    discovery_db = Chroma(
        collection_name="seva_discovery_v1",
        persist_directory=str(VECTORSTORE_DIR / "discovery"),
        embedding_function=embeddings,
    )

    detail_db = Chroma(
        collection_name="seva_detail_v1",
        persist_directory=str(VECTORSTORE_DIR / "detail"),
        embedding_function=embeddings,
    )

    discovery_count = discovery_db._collection.count()
    detail_count = detail_db._collection.count()

    logger.info(
        "Vector stores loaded successfully | Discovery: %s | Detail: %s",
        discovery_count,
        detail_count,
    )

    return discovery_db, detail_db
