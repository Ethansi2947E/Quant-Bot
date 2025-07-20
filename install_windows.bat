@echo off
setlocal

:: ============================================================================
:: Trading Bot Windows Installer
:: ============================================================================
:: This script automates the setup of the trading bot on a Windows system.
:: It performs the following steps:
:: 1. Checks for a Python installation.
:: 2. Creates a virtual environment in the ".venv" directory.
:: 3. Detects the Python version and system architecture.
:: 4. Downloads the correct TA-Lib wheel from Christoph Gohlke's repository.
:: 5. Installs the TA-Lib wheel.
:: 6. Installs all other dependencies from requirements.txt.
:: ============================================================================

echo =================================================
echo  Trading Bot Environment Setup for Windows      
echo =================================================
echo.

:: --- Step 1: Check for Python ---
echo [1/6] Checking for Python installation...
python --version >nul 2>nul
if %errorlevel% neq 0 (
    echo ERROR: Python is not found in your system's PATH.
    echo Please install Python 3.8 or higher and ensure it's added to your PATH.
    pause
    exit /b 1
)
echo Python found.
echo.

:: --- Step 2: Create Virtual Environment ---
echo [2/6] Setting up virtual environment...
if not exist .venv (
    echo Creating virtual environment in ".venv"...
    python -m venv .venv
    if %errorlevel% neq 0 (
        echo ERROR: Failed to create the virtual environment.
        pause
        exit /b 1
    )
) else (
    echo Virtual environment ".venv" already exists.
)
echo.

:: --- Activate Virtual Environment ---
call .venv\Scripts\activate.bat

:: --- Step 3: Get Python Version and Architecture ---
echo [3/6] Detecting Python version and system architecture...
for /f "tokens=*" %%i in ('python -c "import sys; print(f'{sys.version_info.major}{sys.version_info.minor}')"') do set PY_VERSION=%%i
for /f "tokens=*" %%i in ('python -c "import platform; print(platform.architecture()[0])"') do set ARCH=%%i

if "%ARCH%"=="64bit" (
    set ARCH_TAG=win_amd64
) else (
    set ARCH_TAG=win32
)

echo Python version code: cp%PY_VERSION%
echo Architecture: %ARCH_TAG%
echo.

:: --- Step 4: Download the correct TA-Lib wheel ---
echo [4/6] Downloading TA-Lib...
set TA_LIB_VERSION=0.6.4
set TA_LIB_WHEEL=ta_lib-%TA_LIB_VERSION%-cp%PY_VERSION%-cp%PY_VERSION%-%ARCH_TAG%.whl
set DOWNLOAD_URL=https://github.com/cgohlke/talib-build/releases/download/v%TA_LIB_VERSION%/%TA_LIB_WHEEL%

echo Downloading from: %DOWNLOAD_URL%
powershell -Command "Invoke-WebRequest -Uri %DOWNLOAD_URL% -OutFile %TA_LIB_WHEEL%"
if %errorlevel% neq 0 (
    echo ERROR: Failed to download TA-Lib.
    echo Please check your internet connection or manually download the file from the URL above.
    pause
    exit /b 1
)
echo TA-Lib downloaded successfully.
echo.

:: --- Step 5: Install TA-Lib ---
echo [5/6] Installing TA-Lib...
pip install %TA_LIB_WHEEL%
if %errorlevel% neq 0 (
    echo ERROR: Failed to install TA-Lib from the wheel file.
    del %TA_LIB_WHEEL% >nul 2>nul
    pause
    exit /b 1
)
:: Clean up the wheel file after installation
del %TA_LIB_WHEEL%
echo TA-Lib installed successfully.
echo.

:: --- Step 6: Install other dependencies ---
echo [6/6] Installing dependencies from requirements.txt...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Failed to install dependencies from requirements.txt.
    pause
    exit /b 1
)
echo All dependencies installed.
echo.

:: --- Final Message ---
echo =================================================
echo  Setup Complete!                                
echo =================================================
echo.
echo To run the bot, you can now use the run_bot.bat file.
echo.
pause
exit /b 0 