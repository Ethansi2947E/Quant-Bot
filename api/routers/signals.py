from fastapi import APIRouter, Depends
from typing import List
from sqlalchemy.orm import Session
from .. import crud, schemas
from ..database import get_db

router = APIRouter(
    prefix="/api/signals",
    tags=["signals"],
)

@router.get("/", response_model=List[schemas.Signal])
def read_signals(skip: int = 0, limit: int = 20, db: Session = Depends(get_db)):
    """
    Endpoint to get a paginated list of the most recent trading signals.
    """
    signals = crud.get_signals(db, skip=skip, limit=limit)
    return signals 