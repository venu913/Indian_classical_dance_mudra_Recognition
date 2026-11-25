@echo off
title Indian Classical Dance Mudra AI Launcher
echo Starting Backend and Frontend...
start "Kathak Mudra API Backend" cmd /k "cd /d S:\urop_project\UROP_FINAL_2\backend && python app.py"
start "Kathak Mudra React Frontend" cmd /k "cd /d S:\urop_project\UROP_FINAL_2\frontend && npm run dev"
timeout /t 4 /nobreak >nul
start http://localhost:5173
echo Both servers started!
