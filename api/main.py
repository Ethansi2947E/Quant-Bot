import sys
import os

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .database import engine, Base
from .routers import dashboard, signals, active_trades, history, win_loss, performance

# Create all database tables
Base.metadata.create_all(bind=engine)

app = FastAPI()

# Include all the routers
app.include_router(dashboard.router)
app.include_router(signals.router)
app.include_router(active_trades.router)
app.include_router(history.router)
app.include_router(win_loss.router)
app.include_router(performance.router)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this to your frontend's domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Quant-Dash API"} 