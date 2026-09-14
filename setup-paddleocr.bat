@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run setup.bat first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m pip install -r requirements/paddleocr.txt
if errorlevel 1 goto failed
rem RapidOCR requires opencv-python, PaddleX requires opencv-contrib-python.
rem Both publish cv2; install the same-version contrib build last.
".venv\Scripts\python.exe" -m pip install --force-reinstall --no-deps opencv-contrib-python==4.10.0.84
if errorlevel 1 goto failed
".venv\Scripts\python.exe" tools/setup_ocr_models.py
if errorlevel 1 goto failed
".venv\Scripts\python.exe" tools/check_environment.py
echo PaddleOCR is ready. Restart the server using run.bat.
pause
exit /b 0
:failed
echo Setup failed. Check the error above and retry.
pause
exit /b 1
