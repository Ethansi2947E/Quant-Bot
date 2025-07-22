from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from .. import crud, schemas
from ..database import get_db
from ..mt5_service import mt5_service

router = APIRouter(
    prefix="/api/dashboard",
    tags=["dashboard"],
)

@router.get("/")
def read_dashboard_data(db: Session = Depends(get_db)):
    """
    Endpoint to get all data required for the main dashboard page.
    """
    kpis = crud.get_dashboard_kpis(db)
    overview_chart = crud.get_overview_chart_data(db)
    performance_chart = crud.get_performance_chart_data(db)
    
    return {
        "kpis": kpis,
        "overviewChart": overview_chart,
        "performanceChart": performance_chart
    }

@router.get("/account-info")
def get_live_account_info():
    """
    Endpoint to get live account information (balance, equity) from MT5.
    """
    account_info = mt5_service.get_account_info()
    if account_info:
        return {
            "balance": account_info.get("balance"),
            "equity": account_info.get("equity"),
            "profit": account_info.get("profit")
        }
    return {"error": "Could not retrieve account information"} 