from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from typing import Optional
from .. import crud
from ..database import get_db

router = APIRouter(
    prefix="/api/win-loss",
    tags=["win-loss"],
)

@router.get("/")
def read_win_loss_data(
    db: Session = Depends(get_db),
    asset: Optional[str] = Query(None, description="Asset symbol (e.g., 'BTC/USD')"),
    start_date: Optional[str] = Query(None, description="Start date for filtering (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date for filtering (YYYY-MM-DD)"),
):
    """
    Retrieves all data for the Win/Loss Analysis page, with optional filters.
    """
    filters = {
        "asset": asset,
        "start_date": start_date,
        "end_date": end_date,
    }
    # Note: We need a new crud function to get filtered win/loss data.
    # For now, we will call the existing one and ignore filters.
    # This will be updated once we refactor the crud function.
    win_loss_data = crud.get_win_loss_analysis(db, filters=filters)
    
    return win_loss_data 