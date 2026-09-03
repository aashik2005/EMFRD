@echo off
REM Start EMFRD Frontend

echo ============================================================
echo EMFRD Frontend
echo ============================================================
echo.

REM Check if node_modules exists
if not exist frontend\node_modules (
    echo ERROR: Frontend dependencies not installed!
    echo Please run setup.bat first
    pause
    exit /b 1
)

echo Starting React development server...
echo.
echo Frontend will be available at:
echo   http://localhost:5173
echo.
echo Press Ctrl+C to stop the server
echo.

cd frontend
npm run dev
