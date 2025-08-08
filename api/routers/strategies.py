from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from typing import Optional, Dict, Any

from .. import crud
from ..database import get_db

router = APIRouter(
    prefix="/api/strategies",
    tags=["strategies"],
)

@router.get("/")
def get_strategies_overview(
    db: Session = Depends(get_db),
    asset: Optional[str] = Query(None, description="Asset symbol (e.g., 'BTC/USD')"),
    start_date: Optional[str] = Query(None, description="Start date for filtering (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date for filtering (YYYY-MM-DD)"),
):
    """
    Retrieves an overview of all strategies, including aggregated performance
    metrics, with optional filters for asset and date range.
    """
    filters = {
        "asset": asset,
        "start_date": start_date,
        "end_date": end_date,
    }
    strategies_data = crud.get_strategies_overview(db, filters=filters)
    available_assets = crud.get_available_assets(db)
    strategies_data["availableAssets"] = ["All Assets"] + available_assets
    return strategies_data

@router.get("/{strategy_name}")
def get_strategy_details(
    strategy_name: str,
    db: Session = Depends(get_db),
    asset: Optional[str] = Query(None, description="Asset symbol (e.g., 'BTC/USD')"),
    start_date: Optional[str] = Query(None, description="Start date for filtering (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date for filtering (YYYY-MM-DD)"),
):
    """
    Retrieves detailed performance metrics, equity curve, and a list of
    all trades for a single, specified strategy.
    """
    filters = {
        "asset": asset,
        "start_date": start_date,
        "end_date": end_date,
    }
    return crud.get_strategy_details(db, strategy=strategy_name, filters=filters)

