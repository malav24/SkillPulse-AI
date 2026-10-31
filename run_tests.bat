@echo off
REM SkillPulse AI - run all automated tests (Windows).
cd /d "%~dp0backend"
call .venv\Scripts\activate.bat
python -m pytest -q
cd ..\frontend
call npm test
pause
