from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
import json
from pathlib import Path
from loguru import logger
import os
import re

from config.config import (
    TRADING_CONFIG,
    SESSION_CONFIG,
    SIGNAL_THRESHOLDS,
    BASE_DIR
)

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ConfigUpdate(BaseModel):
    trading_config: Dict
    session_config: Dict
    signal_thresholds: Dict

def update_config_section(content: str, section_name: str, new_config: Dict) -> str:
    """Update a specific section in the config file using regex pattern matching."""
    try:
        # Create pattern that matches the entire dictionary definition
        pattern = f"{section_name} = {{[^}}]*}}"
        
        # Format the new config as a proper Python dictionary string
        new_section = f"{section_name} = {{\n"
        for key, value in new_config.items():
            if isinstance(value, str):
                new_section += f'    "{key}": "{value}",\n'
            elif isinstance(value, (list, tuple)):
                new_section += f'    "{key}": {value},\n'
            else:
                new_section += f'    "{key}": {value},\n'
        new_section += "}"
        
        # Replace the old section with the new one
        updated_content = re.sub(pattern, new_section, content, flags=re.DOTALL)
        return updated_content
    except Exception as e:
        logger.error(f"Error updating config section {section_name}: {str(e)}")
        raise

def update_config_file(config_updates: Dict) -> None:
    """Update the configuration file with new values."""
    config_file = BASE_DIR / "config" / "config.py"
    
    if not config_file.exists():
        raise HTTPException(status_code=404, detail="Configuration file not found")
    
    try:
        # Read the current config file
        with open(config_file, "r") as f:
            config_content = f.read()
        
        # Update each section if present in updates
        if "trading_config" in config_updates:
            new_trading_config = {**TRADING_CONFIG, **config_updates["trading_config"]}
            config_content = update_config_section(
                config_content, 
                "TRADING_CONFIG",
                new_trading_config
            )
        
        if "session_config" in config_updates:
            new_session_config = {**SESSION_CONFIG, **config_updates["session_config"]}
            config_content = update_config_section(
                config_content,
                "SESSION_CONFIG",
                new_session_config
            )
        
        if "signal_thresholds" in config_updates:
            new_signal_thresholds = {**SIGNAL_THRESHOLDS, **config_updates["signal_thresholds"]}
            config_content = update_config_section(
                config_content,
                "SIGNAL_THRESHOLDS",
                new_signal_thresholds
            )
        
        # Write the updated config back to file
        with open(config_file, "w") as f:
            f.write(config_content)
            
        logger.info("Configuration updated successfully")
        
    except Exception as e:
        logger.error(f"Error updating configuration: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update configuration: {str(e)}")

@app.post("/api/update-config")
async def update_config(config: ConfigUpdate):
    """Update the trading bot configuration."""
    try:
        update_config_file(config.dict())
        
        return {
            "success": True,
            "message": "Configuration updated successfully"
        }
        
    except Exception as e:
        logger.error(f"Error in update_config endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/config")
async def get_config():
    """Get the current trading bot configuration."""
    try:
        return {
            "trading_config": TRADING_CONFIG,
            "session_config": SESSION_CONFIG,
            "signal_thresholds": SIGNAL_THRESHOLDS
        }
    except Exception as e:
        logger.error(f"Error in get_config endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 