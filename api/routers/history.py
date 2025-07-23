from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from typing import Optional
from .. import crud, schemas
from ..database import get_db

router = APIRouter(
    prefix="/api/history",
    tags=["history"],
)

@router.get("/", response_model=schemas.TradeHistoryResponse)
def read_trade_history(
    db: Session = Depends(get_db),
    page: int = Query(1, ge=1, description="Page number for pagination"),
    page_size: int = Query(20, ge=1, le=100, description="Number of trades per page"),
    sort_by: Optional[str] = Query("close_time", description="Column to sort by"),
    sort_order: Optional[str] = Query("desc", description="Sort order (asc or desc)"),
    search: Optional[str] = Query(None, description="Search term for symbol or type")
):
    """
    Retrieves a paginated, sorted, and searchable list of trade history.
    """
    skip = (page - 1) * page_size
    history_data = crud.get_trade_history(
        db, 
        skip=skip, 
        limit=page_size, 
        sort_by=sort_by, 
        sort_order=sort_order, 
        search=search
    )
    return history_data 