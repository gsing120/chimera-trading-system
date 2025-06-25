@echo off
REM Chimera Trading System v2.0 - Windows Startup Script

echo ========================================
echo Chimera Trading System v2.0
echo ========================================

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.7+ and try again
    pause
    exit /b 1
)

REM Check if we're in the right directory
if not exist "main.py" (
    echo ERROR: main.py not found
    echo Please run this script from the chimera_trading_system directory
    pause
    exit /b 1
)

REM Install dependencies if needed (optional, system works without them)
echo Checking dependencies...
python -c "import numpy, sklearn" >nul 2>&1
if errorlevel 1 (
    echo Installing optional dependencies...
    python -m pip install -r requirements.txt
)

REM Run the system
echo Starting Chimera Trading System...
echo.
python main.py %*

if errorlevel 1 (
    echo.
    echo System exited with error code %errorlevel%
    pause
)

echo.
echo System shutdown complete.
pause

