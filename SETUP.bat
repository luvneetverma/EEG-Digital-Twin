@echo off
title EEG Digital Twin — Setup
color 0A
cd /d "%~dp0"

echo.
echo ================================================
echo   EEG Digital Twin — Local Setup
echo   IEEE: Cloud-Based EEG Digital Twin Framework
echo ================================================
echo.
echo Folder: %CD%
echo.

:: Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
  echo ERROR: Python not found. Install from https://python.org
  pause & exit /b 1
)

:: Create virtual environment if missing
if not exist "venv\Scripts\activate.bat" (
  echo [1/4] Creating virtual environment...
  python -m venv venv
)

:: Activate venv
call venv\Scripts\activate.bat

echo [2/4] Installing dependencies...
pip install -r requirements.txt --quiet
if %errorlevel% neq 0 (
  echo ERROR: pip install failed.
  pause & exit /b 1
)

echo [3/4] Training ML models (SVM, RF, LSTM proxy)...
python train_models.py
if %errorlevel% neq 0 (
  echo ERROR: Training failed. See above for details.
  pause & exit /b 1
)

echo.
echo [4/4] Starting EEG Digital Twin API + Dashboard...
echo.
echo ================================================
echo   Dashboard: http://127.0.0.1:5000
echo   Health   : http://127.0.0.1:5000/health
echo   Demo     : http://127.0.0.1:5000/twin/demo
echo   Docs     : http://127.0.0.1:5000/docs
echo ================================================
echo.
echo   Opening browser in 3 seconds...
timeout /t 3 /nobreak >nul
start http://127.0.0.1:5000

python api.py
pause
