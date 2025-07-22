@echo off
::
:: Robust Update Script for Trading Bot (Windows)
::
:: This script safely updates the local repository with the latest changes
:: from the 'SuperTrend' branch on GitHub. It also intelligently handles
:: dependency updates for both the Python backend and the Next.js frontend.
::

echo 🚀 Starting the update process...

:: --- Check for local changes and stash them ---
for /f "tokens=*" %%a in ('git status --porcelain') do (
    echo ⚠️  Local changes detected. Stashing them...
    git stash push -m "Auto-stashed by update script"
    set STASHED=true
    goto :StashDone
)
set STASHED=false
:StashDone

:: --- Remember current branch ---
for /f "tokens=*" %%a in ('git rev-parse --abbrev-ref HEAD') do set ORIGINAL_BRANCH=%%a
echo 🔄 Current branch is '%ORIGINAL_BRANCH%'. Switching to 'SuperTrend' for update.

:: --- Fetch and Pull ---
git checkout SuperTrend
echo 🚚 Pulling latest changes from origin/SuperTrend...
git pull origin SuperTrend

:: --- Check for Python dependency changes ---
git diff HEAD@{1}..HEAD --quiet -- requirements.txt
if %errorlevel% equ 0 (
    echo 🐍 No changes in Python dependencies.
) else (
    echo 🐍 Python dependencies have changed. Installing...
    if exist .venv\\Scripts\\activate.bat (
        call .venv\\Scripts\\activate.bat
    )
    pip install -r requirements.txt
)

:: --- Check for Node.js dependency changes ---
git diff HEAD@{1}..HEAD --quiet -- Quant-Dash/package.json
if %errorlevel% equ 0 (
    echo 📦 No changes in Node.js dependencies.
) else (
    echo 📦 Node.js dependencies have changed. Installing...
    cd Quant-Dash
    pnpm install
    cd ..
)

:: --- Return to original branch and restore changes ---
echo 🔄 Returning to original branch '%ORIGINAL_BRANCH%'...
git checkout %ORIGINAL_BRANCH%

if "%STASHED%"=="true" (
    echo  Restoring local changes...
    git stash pop
)

echo ✅ Update complete! Your trading bot is now up-to-date.
pause 