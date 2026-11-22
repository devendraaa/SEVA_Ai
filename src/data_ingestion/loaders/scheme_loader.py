import json
from pathlib import Path
from core.logging import logger

DATA_PATH = Path(r"G:\SEVA_Ai\SEVA_Ai\notebooks\data_analysis\all_schemes.json")

def load_schemes(file_path: Path = DATA_PATH) -> list[dict]:

    logger.info("Starting scheme data ingestion")

    try:
        logger.info("Loading dataset: %s", file_path)

        with file_path.open("r", encoding="utf-8") as file:
            schemes = json.load(file)

        if not isinstance(schemes, list):
            raise ValueError("Expected a list of scheme records")

        logger.info(
            "Successfully loaded %d schemes",
            len(schemes),
        )

        return schemes

    except Exception:
        logger.exception("Scheme data ingestion failed")
        raise