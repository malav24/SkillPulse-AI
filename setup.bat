@echo off
REM ==========================================================================
REM  SkillPulse AI - one-time setup (Windows). Double-click this file.
REM  Needs: Python 3.11, 3.12 or 3.13  and  Node.js 18+ (both on PATH).
REM ==========================================================================
setlocal
cd /d "%~dp0"
echo.
echo === SkillPulse AI setup ===
echo.

REM ---- find Python (3.12 preferred) ----
set "PY="
for %%v in (3.12 3.13 3.11) do (
  if not defined PY (
    py -%%v -c "import sys" >nul 2>&1 && set "PY=py -%%v"
  )
)
if not defined PY (
  python -c "import sys; assert (3,11) <= sys.version_info[:2] <= (3,13)" >nul 2>&1 && set "PY=python"
)
if not defined PY (
  echo [ERROR] Python 3.11, 3.12 or 3.13 was not found.
  echo         Install Python 3.12 from https://www.python.org/downloads/  ^(tick "Add python.exe to PATH"^)
  goto :fail
)
echo Using Python: %PY%

where npm >nul 2>&1
if errorlevel 1 (
  echo [ERROR] Node.js was not found. Install Node.js 20 LTS from https://nodejs.org/
  goto :fail
)

REM ---- backend ----
echo.
echo [1/3] Creating the Python virtual environment in backend\.venv ...
cd backend
if not exist ".venv\Scripts\python.exe" (
  %PY% -m venv .venv || goto :fail
)
call .venv\Scripts\activate.bat
echo [2/3] Installing Python packages (first time: about 5 minutes) ...
python -m pip install --upgrade pip >nul
pip install -r requirements-dev.txt || goto :fail
if not exist ".env" copy ".env.example" ".env" >nul
cd ..

REM ---- frontend ----
echo.
echo [3/3] Installing frontend packages (first time: about 2 minutes) ...
cd frontend
call npm install || goto :fail
cd ..

echo.
echo ==========================================================
echo  Setup complete.  Now double-click  start.bat
echo ==========================================================
pause
exit /b 0

:fail
echo.
echo Setup did not finish. Read the error above, fix it, then run setup.bat again.
pause
exit /b 1
