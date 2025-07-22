from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from .. import crud
from ..database import get_db

router = APIRouter(
    prefix="/api/win-loss",
    tags=["win-loss"],
)

@router.get("/")
def read_win_loss_data(db: Session = Depends(get_db)):
    """
    Endpoint to get all data required for the Win/Loss Analysis page.
    """
    return crud.get_win_loss_analysis(db) 