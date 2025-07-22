from fastapi import APIRouter, Depends
from typing import List, Optional, Dict, Any

# Adjust the import path to use the new service singleton
from ..mt5_service import mt5_service

router = APIRouter(
    prefix="/api/active-trades",
    tags=["active-trades"],
)

@router.get("/", response_model=List[Dict[str, Any]])
def get_live_active_trades(magic_number: Optional[int] = None):
    """
    Endpoint to fetch and return a list of currently active trades
    directly from the MT5 terminal.
    
    An optional `magic_number` can be provided to filter trades.
    """
    active_trades = mt5_service.get_active_trades(magic_number=magic_number)
    return active_trades 