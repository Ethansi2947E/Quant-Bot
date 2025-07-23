from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from typing import Optional
from .. import crud
from ..database import get_db

router = APIRouter(
    prefix="/api/performance",
    tags=["performance"],
)

@router.get("/")
def read_performance_data(
    db: Session = Depends(get_db),
    asset: Optional[str] = Query(None, description="Asset symbol (e.g., 'BTC/USD')"),
    start_date: Optional[str] = Query(None, description="Start date for filtering (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date for filtering (YYYY-MM-DD)"),
    timeframe: str = Query("1Y", description="Timeframe for performance chart (e.g., 1W, 1M, 1Y)")
):
    """
    Endpoint to get all data required for the performance analytics page.
    This includes data for the Overview, Risk Analysis, and Monthly Breakdown tabs.
    """
    filters = {
        "asset": asset,
        "start_date": start_date,
        "end_date": end_date,
        "timeframe": timeframe
    }
    
    overview_data = crud.get_performance_overview(db, filters=filters)
    risk_analysis_data = crud.get_risk_analysis(db, filters=filters)
    monthly_breakdown_data = crud.get_monthly_breakdown(db, filters=filters)
    available_assets = crud.get_available_assets(db)
    
    return {
        "filters": {
            "assets": ["All Assets"] + available_assets,
            "dateRange": { "min": "2024-01-01", "max": "2024-12-31" } # Placeholder for now
        },
        "overview": overview_data,
        "riskAnalysis": risk_analysis_data,
        "monthlyBreakdown": monthly_breakdown_data
    } 