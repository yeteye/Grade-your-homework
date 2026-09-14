@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  py -3.12 -m venv .venv
  if errorlevel 1 (
    echo Install Python 3.12 from python.org, then run setup.bat again.
    pause
    exit /b 1
  )
)
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
  pause
  exit /b 1
)
echo Basic setup complete. Use setup-ai.bat for offline OCR and local model support.
pause
