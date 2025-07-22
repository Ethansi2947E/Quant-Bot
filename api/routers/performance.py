from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from .. import crud
from ..database import get_db

router = APIRouter(
    prefix="/api/performance",
    tags=["performance"],
)

@router.get("/")
def read_performance_data(db: Session = Depends(get_db)):
    """
    Endpoint to get all data required for the performance analytics page.
    This includes data for the Overview, Risk Analysis, and Monthly Breakdown tabs.
    """
    overview_data = crud.get_performance_overview(db)
    risk_analysis_data = crud.get_risk_analysis(db)
    monthly_breakdown_data = crud.get_monthly_breakdown(db)

    # Note: Filtering by asset and date range is not yet implemented.
    # The API will currently return data for all assets.
    
    return {
        "filters": {
            "assets": ["All Assets", "BTC/USD", "ETH/USD"], # Placeholder
            "dateRange": { "min": "2024-01-01", "max": "2024-12-31" } # Placeholder
        },
        "overview": overview_data,
        "riskAnalysis": risk_analysis_data,
        "monthlyBreakdown": monthly_breakdown_data
    } 