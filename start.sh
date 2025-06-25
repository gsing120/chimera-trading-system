#!/bin/bash
# Chimera Trading System v2.0 - Unix/Linux/Mac Startup Script

echo "========================================"
echo "Chimera Trading System v2.0"
echo "========================================"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    if ! command -v python &> /dev/null; then
        echo "ERROR: Python is not installed or not in PATH"
        echo "Please install Python 3.7+ and try again"
        exit 1
    else
        PYTHON_CMD="python"
    fi
else
    PYTHON_CMD="python3"
fi

# Check if we're in the right directory
if [ ! -f "main.py" ]; then
    echo "ERROR: main.py not found"
    echo "Please run this script from the chimera_trading_system directory"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$($PYTHON_CMD -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
REQUIRED_VERSION="3.7"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "ERROR: Python $REQUIRED_VERSION or higher is required"
    echo "Current version: $PYTHON_VERSION"
    exit 1
fi

# Install dependencies if needed (optional, system works without them)
echo "Checking dependencies..."
$PYTHON_CMD -c "import numpy, sklearn" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing optional dependencies..."
    $PYTHON_CMD -m pip install -r requirements.txt
fi

# Make sure the script is executable
chmod +x "$0"

# Run the system
echo "Starting Chimera Trading System..."
echo ""
$PYTHON_CMD main.py "$@"

EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo ""
    echo "System exited with error code $EXIT_CODE"
fi

echo ""
echo "System shutdown complete."

