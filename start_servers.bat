@echo off
echo ========================================================
echo   HOTEL CANCELLATION ML PLATFORM - FLASK + ANGULAR
echo ========================================================
echo.

echo Starting Flask API Backend on http://127.0.0.1:5000 ...
start "Flask API" cmd /k "python backend/app.py"

timeout /t 2 /nobreak >nul

echo Starting Angular Frontend on http://localhost:4200 ...
cd frontend
start "Angular Frontend" cmd /k "npm start -- --port 4200"

echo.
echo ========================================================
echo   Both services are launching!
echo   Frontend: http://localhost:4200
echo   Backend:  http://127.0.0.1:5000
echo ========================================================
pause
