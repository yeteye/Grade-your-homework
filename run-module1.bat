@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run setup.bat first to create the test environment.
  exit /b 2
)
".venv\Scripts\python.exe" tools\run_module1.py %*
exit /b %errorlevel%
