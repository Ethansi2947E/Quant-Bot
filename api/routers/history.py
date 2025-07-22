from fastapi import APIRouter, Depends
from typing import List, Optional
from sqlalchemy.orm import Session
from .. import crud, schemas
from ..database import get_db

router = APIRouter(
    prefix="/api/history",
    tags=["history"],
)

@router.get("/")
def read_trade_history(
    skip: int = 0, 
    limit: int = 20, 
    sort_by: str = "close_time",
    sort_order: str = "desc",
    search: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    Endpoint to get a paginated, sorted, and searchable list of trade history.
    """
    history_data = crud.get_trade_history(db, skip=skip, limit=limit, sort_by=sort_by, sort_order=sort_order, search=search)
    return history_data 