#!/bin/bash

# ============================================================================
# Trading Bot Linux/macOS Installer
# ============================================================================
# This script automates the setup of the trading bot on a Linux or macOS system.
# It performs the following steps:
# 1. Checks for Python 3.
# 2. Creates a virtual environment in ".venv".
# 3. Installs the TA-Lib C library using the system's package manager.
# 4. Installs the TA-Lib Python wrapper and other dependencies.
# ============================================================================

echo "================================================="
echo " Trading Bot Environment Setup for Linux/macOS "
echo "================================================="
echo

# --- Step 1: Check for Python ---
echo "[1/5] Checking for Python 3 installation..."
if ! command -v python3 &> /dev/null; then
    echo "ERROR: python3 could not be found."
    echo "Please install Python 3.8 or higher to continue."
    exit 1
fi
echo "Python 3 found."
echo

# --- Step 2: Create Virtual Environment ---
echo "[2/5] Setting up virtual environment..."
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment in '.venv'..."
    python3 -m venv .venv
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to create the virtual environment."
        exit 1
    fi
else
    echo "Virtual environment '.venv' already exists."
fi
echo

# --- Activate Virtual Environment ---
source .venv/bin/activate

# --- Step 3: Install TA-Lib C library ---
echo "[3/5] Installing TA-Lib C library..."
# Check for macOS (Homebrew)
if [[ "$OSTYPE" == "darwin"* ]]; then
    if command -v brew &> /dev/null; then
        echo "Detected macOS. Installing TA-Lib with Homebrew..."
        brew install ta-lib
    else
        echo "ERROR: Homebrew not found. Please install Homebrew or install TA-Lib manually."
        exit 1
    fi
# Check for Debian/Ubuntu (apt)
elif command -v apt-get &> /dev/null; then
    echo "Detected Debian/Ubuntu. Installing with apt-get..."
    echo "You may be prompted for your password to install system packages."
    sudo apt-get update
    sudo apt-get install -y libta-lib-dev
# Check for Fedora/CentOS/RHEL (dnf/yum)
elif command -v dnf &> /dev/null; then
    echo "Detected Fedora/RHEL. Installing with dnf..."
    echo "You may be prompted for your password to install system packages."
    sudo dnf install -y ta-lib-devel
elif command -v yum &> /dev/null; then
    echo "Detected CentOS/RHEL. Installing with yum..."
    echo "You may be prompted for your password to install system packages."
    sudo yum install -y ta-lib-devel
else
    echo "WARNING: Could not detect package manager (apt, dnf, yum, or brew)."
    echo "Please install the TA-Lib C library manually for your system."
    echo "Press Enter to try and continue the installation without it, or Ctrl+C to exit."
    read -r
fi

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install the TA-Lib C library. Please try installing it manually."
    exit 1
fi
echo "TA-Lib C library installed."
echo

# --- Step 4: Install TA-Lib Python Wrapper ---
echo "[4/5] Installing TA-Lib Python package..."
pip install TA-Lib
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install the TA-Lib Python wrapper."
    echo "This usually happens if the C library was not installed correctly in the previous step."
    exit 1
fi
echo "TA-Lib Python package installed."
echo

# --- Step 5: Install other dependencies ---
echo "[5/5] Installing dependencies from requirements.txt..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install dependencies from requirements.txt."
    exit 1
fi
echo "All dependencies installed."
echo

# --- Final Message ---
echo "================================================="
echo " Setup Complete!                                "
echo "================================================="
echo
echo "To activate the environment in the future, run:"
echo "source .venv/bin/activate"
echo
echo "To run the bot, use:"
echo "python3 main.py"
echo 