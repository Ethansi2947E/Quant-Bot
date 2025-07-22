import sys
import os

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routers import dashboard, performance, win_loss, history, active_trades, signals
from .database import engine, Base
from . import models

# Create all tables in the database
# This will check for the existence of tables and create them if they don't exist
# It will now include the new 'signals' table.
models.Base.metadata.create_all(bind=engine)

app = FastAPI()

# Set up CORS
origins = [
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(dashboard.router)
app.include_router(performance.router)
app.include_router(win_loss.router)
app.include_router(history.router)
app.include_router(active_trades.router)
app.include_router(signals.router)

@app.get("/")
def read_root():
    return {"message": "Trading Bot API is running"} 