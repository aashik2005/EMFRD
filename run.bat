@echo off
REM Run EMFRD - Starts both backend and frontend

echo ============================================================
echo EMFRD - Starting Complete System
echo ============================================================
echo.

REM Check if setup was run
if not exist venv (
    echo ERROR: Setup not complete!
    echo Please run setup.bat first
    echo.
    pause
    exit /b 1
)

if not exist frontend\node_modules (
    echo ERROR: Frontend dependencies not installed!
    echo Please run setup.bat first
    echo.
    pause
    exit /b 1
)

echo Starting backend server in new window...
start "EMFRD Backend" cmd /k start_backend.bat

echo Waiting 5 seconds for backend to start...
timeout /t 5 /nobreak >nul

echo Starting frontend in new window...
start "EMFRD Frontend" cmd /k start_frontend.bat

echo.
echo ============================================================
echo EMFRD is starting!
echo ============================================================
echo.
echo Two windows will open:
echo   1. Backend (API): http://localhost:8000
echo   2. Frontend (UI): http://localhost:5173
echo.
echo Wait ~10 seconds, then open: http://localhost:5173
echo.
echo To stop: Close both terminal windows
echo.
pause
