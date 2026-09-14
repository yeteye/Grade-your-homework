@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run setup.bat first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m pip install -r requirements/ocr.txt -r requirements/model.txt
if errorlevel 1 (
  pause
  exit /b 1
)
echo Offline OCR and local model dependencies installed.
pause
