#!/bin/bash
#
# Robust Update Script for Trading Bot
#
# This script safely updates the local repository with the latest changes
# from the 'SuperTrend' branch on GitHub. It also intelligently handles
# dependency updates for both the Python backend and the Next.js frontend.
#

set -e # Exit immediately if a command exits with a non-zero status.

echo "🚀 Starting the update process..."

# --- Check for clean working directory ---
if ! git diff-index --quiet HEAD --; then
    echo "⚠️  Local changes detected. Stashing them..."
    git stash push -m "Auto-stashed by update script"
    STASHED=true
else
    STASHED=false
fi

# --- Remember current branch ---
ORIGINAL_BRANCH=$(git rev-parse --abbrev-ref HEAD)
echo "🔄 Current branch is '$ORIGINAL_BRANCH'. Switching to 'SuperTrend' for update."

# --- Fetch and Pull ---
git checkout SuperTrend
echo "🚚 Pulling latest changes from origin/SuperTrend..."
git pull origin SuperTrend

# --- Check for Python dependency changes ---
if git diff HEAD@{1}..HEAD --quiet -- requirements.txt; then
    echo "🐍 No changes in Python dependencies."
else
    echo "🐍 Python dependencies have changed. Installing..."
    # Activate virtual environment if it exists
    if [ -d ".venv" ]; then
        source .venv/bin/activate
    fi
    pip install -r requirements.txt
fi

# --- Check for Node.js dependency changes ---
if git diff HEAD@{1}..HEAD --quiet -- Quant-Dash/package.json; then
    echo "📦 No changes in Node.js dependencies."
else
    echo "📦 Node.js dependencies have changed. Installing..."
    cd Quant-Dash
    pnpm install
    cd ..
fi

# --- Return to original branch and restore changes ---
echo "🔄 Returning to original branch '$ORIGINAL_BRANCH'..."
git checkout "$ORIGINAL_BRANCH"

if [ "$STASHED" = true ]; then
    echo " restorating local changes..."
    git stash pop
fi

echo "✅ Update complete! Your trading bot is now up-to-date." 