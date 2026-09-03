@echo off
REM EMFRD Setup Script for Windows
REM Automates the complete installation process

echo ============================================================
echo EMFRD - Complete Setup Script
echo ============================================================
echo.

REM Check Python
echo [1/5] Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python not found!
    echo.
    echo Please install Python 3.11 or higher from:
    echo https://www.python.org/downloads/
    echo.
    echo IMPORTANT: Check "Add Python to PATH" during installation
    echo.
    pause
    exit /b 1
)

echo ✓ Python found
python --version
echo.

REM Check Node.js
echo [2/5] Checking Node.js installation...
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Node.js not found!
    echo.
    echo Please install Node.js 18+ from:
    echo https://nodejs.org/
    echo.
    pause
    exit /b 1
)

echo ✓ Node.js found
node --version
echo.

REM Create virtual environment
echo [3/5] Setting up Python virtual environment...
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
    echo ✓ Virtual environment created
) else (
    echo ✓ Virtual environment already exists
)
echo.

REM Install Python dependencies
echo [4/5] Installing Python dependencies...
echo This may take 5-10 minutes...
call venv\Scripts\activate.bat
pip install --upgrade pip
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Failed to install Python dependencies
    pause
    exit /b 1
)
echo ✓ Python dependencies installed
echo.

REM Install frontend dependencies
echo [5/5] Installing frontend dependencies...
echo This may take 2-3 minutes...
cd frontend
call npm install
if %errorlevel% neq 0 (
    echo ERROR: Failed to install frontend dependencies
    cd ..
    pause
    exit /b 1
)
cd ..
echo ✓ Frontend dependencies installed
echo.

echo ============================================================
echo Setup Complete! ✓
echo ============================================================
echo.
echo Next steps:
echo   1. Run: start_backend.bat (to start API server)
echo   2. Run: start_frontend.bat (to start frontend)
echo   3. Open: http://localhost:5173
echo.
echo Or simply run: run.bat (starts both automatically)
echo.
pause
