@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] Missing .venv. Run setup.bat first.
  exit /b 1
)
".venv\Scripts\python.exe" -m unittest checks.test_module2 -v
set "TEST_EXIT=%ERRORLEVEL%"
echo.
if "%TEST_EXIT%"=="0" (echo Module 2 tests passed.) else (echo Module 2 tests failed.)
pause
exit /b %TEST_EXIT%
