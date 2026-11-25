@echo off
title Indian Classical Dance Mudra AI Launcher
echo =====================================================================
echo    Starting Indian Classical Dance Mudra AI System (FastViT-T8)
echo =====================================================================
echo.
echo [1/3] Starting FastAPI Backend on http://localhost:8000 ...
start "Kathak Mudra API Backend" cmd /k "cd /d S:\urop_project\UROP_FINAL_2\backend && python -m uvicorn app:app --host 0.0.0.0 --port 8000"

echo [2/3] Starting React / Vite Frontend on http://localhost:5173 ...
start "Kathak Mudra React Frontend" cmd /k "cd /d S:\urop_project\UROP_FINAL_2\frontend && npm run dev"

echo [3/3] Waiting 4 seconds for services to initialize...
timeout /t 4 /nobreak >nul

echo Opening browser at http://localhost:5173 ...
start http://localhost:5173

echo.
echo =====================================================================
echo    Both servers are running!
echo    - Backend API : http://localhost:8000 (Docs: http://localhost:8000/docs)
echo    - Web Frontend: http://localhost:5173
echo =====================================================================
echo To stop the servers, simply close their respective terminal windows.
pause
