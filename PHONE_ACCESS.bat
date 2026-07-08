@echo off
title AuraScribe Pro - Phone Access Setup
color 0A

echo ==========================================
echo    AuraScribe Pro - Phone Access Setup
echo ==========================================
echo.

REM Get IP address
echo Your PC's IP Address:
echo.
for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /i "IPv4"') do (
    set IP=%%a
    echo    %%a
)
echo.

echo ==========================================
echo    Instructions for Phone Access
echo ==========================================
echo.
echo 1. Make sure your phone and PC are on SAME WiFi
echo.
echo 2. Start the server:
echo    python app.py
echo.
echo 3. Open browser on your phone and go to:
echo    http://%IP%:8000
echo.
echo ==========================================
echo    Or use localhost if running on same PC:
echo    http://localhost:8000
echo ==========================================
echo.
echo Press any key to start server...
pause > nul

REM Start the server
python app.py
