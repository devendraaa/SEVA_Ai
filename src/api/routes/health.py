from fastapi import APIRouter
from fastapi import Depends
from langchain_chroma import Chroma
from core.dependencies import get_discovery_db, get_detail_db

router = APIRouter(tags=["Health"])

@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "application": "SEVA_AI"
    }

@router.get("/vectorstores")
async def check_vectorstores(
    discovery_db: Chroma = Depends(get_discovery_db),
    detail_db: Chroma = Depends(get_detail_db),
):
    return {
        "discovery": discovery_db._collection.count(),
        "detail": detail_db._collection.count(),
        "status": "connected",
    }
