@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] Missing .venv. Run setup.bat first.
  exit /b 1
)
".venv\Scripts\python.exe" -m unittest checks.test_module1 -v
exit /b %ERRORLEVEL%
