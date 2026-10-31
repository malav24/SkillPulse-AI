@echo off
REM ==========================================================================
REM  SkillPulse AI - start the app (Windows). Run setup.bat once before this.
REM  Opens two windows (backend + frontend) and the browser.
REM  To stop: close the two windows (or press Ctrl+C in each).
REM ==========================================================================
cd /d "%~dp0"
if not exist "backend\.venv\Scripts\activate.bat" (
  echo Run setup.bat first.
  pause
  exit /b 1
)
start "SkillPulse backend (port 8000)" cmd /k "cd /d "%~dp0backend" && call .venv\Scripts\activate.bat && uvicorn app.main:app --reload --port 8000"
start "SkillPulse frontend (port 5173)" cmd /k "cd /d "%~dp0frontend" && npm run dev"
echo Starting SkillPulse ... the browser opens in a few seconds.
timeout /t 10 /nobreak >nul
start "" http://localhost:5173
