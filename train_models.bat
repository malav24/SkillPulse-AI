@echo off
REM ==========================================================================
REM  SkillPulse AI - retrain all 3 machine-learning models (Windows).
REM  Takes about 10 minutes. Results go to backend\artifacts\ (old ones are
REM  copied to backend\artifacts_backup\ first). Do NOT click inside this
REM  window while it runs (that pauses it on Windows).
REM ==========================================================================
cd /d "%~dp0backend"
if not exist ".venv\Scripts\activate.bat" (
  echo Run setup.bat first.
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat
if exist artifacts (
  if exist artifacts_backup rmdir /s /q artifacts_backup
  xcopy /e /i /q artifacts artifacts_backup >nul
)
python -m ml.train_all
echo.
echo Training finished. Restart the backend (close it and run start.bat) to use the new models.
pause
