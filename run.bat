@echo off
title AuraScribe AI launcher
echo ===================================================
echo             AURASCRIBE AI LAUNCHER
echo ===================================================
echo.

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in the PATH.
    echo Please install Python 3.10+ and select "Add to PATH" during installation.
    pause
    exit /b
)

:: Activate/Create Virtual Environment
if not exist "venv" (
    echo [INFO] Virtual environment 'venv' not found. Creating virtual environment...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b
    )
    
    echo [INFO] Installing required dependencies...
    call .\venv\Scripts\activate.bat
    pip install fastapi uvicorn python-multipart faster-whisper python-docx reportlab numpy soundfile
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to install dependencies.
        pause
        exit /b
    )
) else (
    echo [INFO] Activating virtual environment...
    call .\venv\Scripts\activate.bat
)

echo.
echo ===================================================
echo  LOCAL AUDIO TRANSCRIBER & SUMMARIZER ENGINE STARTED
echo ===================================================
echo.
echo  - Voice Recognition Engine: Open-Source Whisper Model
echo  - Server Address: http://127.0.0.1:8000
echo  - Cost: $0 (100%% Free & Local)
echo  - Internet Connection: Optional (Works fully offline!)
echo  - Output Directory: D:\Voice transcriber\transcriptions
echo.
echo ===================================================
echo.
echo Opening browser...
start http://127.0.0.1:8000
echo.
echo Running server...
python app.py

pause
