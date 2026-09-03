@echo off
REM Quick training on demo dataset

echo ============================================================
echo EMFRD - Quick Training (Demo Dataset)
echo ============================================================
echo.
echo This will train RoBERTa on the demo dataset (40 samples)
echo Expected time: 2-5 minutes
echo.
pause

REM Activate virtual environment
call venv\Scripts\activate.bat

echo.
echo [1/1] Training RoBERTa Baseline...
echo ============================================================
python -m backend.training.train_roberta --epochs 5 --batch-size 16

if %errorlevel% equ 0 (
    echo.
    echo ============================================================
    echo Training Complete! ✓
    echo ============================================================
    echo.
    echo Model saved to: models/roberta_baseline/best.pt
    echo Results saved to: experiments/results/
    echo.
    echo Next steps:
    echo   1. Start the system: run.bat
    echo   2. Test predictions in browser
    echo.
) else (
    echo.
    echo ERROR: Training failed!
    echo.
)

pause
