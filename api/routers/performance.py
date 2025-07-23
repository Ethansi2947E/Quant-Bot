from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from .. import crud
from ..database import get_db

router = APIRouter(
    prefix="/api/performance",
    tags=["performance"],
)

@router.get("/")
def read_performance_data(
    db: Session = Depends(get_db),
    timeframe: str = Query("1Y", description="Timeframe for performance chart (e.g., 1W, 1M, 1Y)")
):
    """
    Endpoint to get all data required for the performance analytics page.
    This includes data for the Overview, Risk Analysis, and Monthly Breakdown tabs.
    """
    overview_data = crud.get_performance_overview(db, timeframe=timeframe)
    risk_analysis_data = crud.get_risk_analysis(db)
    monthly_breakdown_data = crud.get_monthly_breakdown(db)

    # The API will now filter performance data based on the timeframe.
    
    return {
        "filters": {
            "assets": ["All Assets", "BTC/USD", "ETH/USD"], # Placeholder
            "dateRange": { "min": "2024-01-01", "max": "2024-12-31" } # Placeholder
        },
        "overview": overview_data,
        "riskAnalysis": risk_analysis_data,
        "monthlyBreakdown": monthly_breakdown_data
    } 