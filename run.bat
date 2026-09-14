@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Environment missing. Run setup.bat first.
  pause
  exit /b 1
)
if not defined PORT set PORT=5000
echo Starting http://127.0.0.1:%PORT%
".venv\Scripts\python.exe" launch.py
if errorlevel 1 pause
