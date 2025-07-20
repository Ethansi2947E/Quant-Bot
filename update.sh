#!/bin/bash

# ============================================================================
# Trading Bot Updater
# ============================================================================
# This script automates the process of updating the trading bot to the latest
# version from the Git repository and rebuilding the Docker image.
# ============================================================================

echo "========================================="
echo "  Starting Trading Bot Update Process  "
echo "========================================="
echo

# --- Step 1: Stop the current bot ---
echo "[1/4] Stopping the currently running bot..."
docker-compose down
echo "Bot stopped."
echo

# --- Step 2: Pull latest changes ---
echo "[2/4] Pulling the latest updates from the Git repository..."
git pull
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to pull updates from Git."
    echo "Please resolve any merge conflicts or stash your local changes and try again."
    exit 1
fi
echo "Successfully pulled the latest code."
echo

# --- Step 3: Rebuild the Docker image ---
echo "[3/4] Rebuilding the Docker image with the latest updates..."
docker-compose build
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to rebuild the Docker image."
    exit 1
fi
echo "Docker image rebuilt successfully."
echo

# --- Step 4: Restart the bot ---
echo "[4/4] Restarting the bot in detached mode..."
docker-compose up -d
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to restart the bot."
    exit 1
fi
echo

# --- Final Message ---
echo "========================================="
echo "  Update Complete!                       "
echo "========================================="
echo
echo "The trading bot has been updated and is now running with the latest version."
echo 