@echo off
echo ============================================================
echo Site Deployment Tool - Starting Server
echo ============================================================
echo.

cd /d "G:\My Drive\Windsor\Mode Server (Original)\site_deployment_demo"

set PYTHON=C:\Users\T773491\AppData\Local\anaconda3\python.exe

echo Checking Python...
"%PYTHON%" --version
echo.

echo Starting Flask server...
echo Open your browser to: http://localhost:5000
echo Press Ctrl+C to stop the server
echo.
echo ============================================================
echo.

"%PYTHON%" app.py

pause
