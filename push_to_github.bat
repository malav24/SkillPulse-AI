@echo off
REM ==========================================================================
REM  Push this project to GitHub as  github.com/malav24/SkillPulse-AI
REM  1. Create an EMPTY repository named SkillPulse-AI on https://github.com/new
REM     (Private recommended; do NOT add a README, .gitignore or licence there)
REM  2. Double-click this file. Git asks you to sign in to GitHub the first time.
REM ==========================================================================
setlocal
cd /d "%~dp0"
set "REPO=https://github.com/malav24/SkillPulse-AI.git"
where git >nul 2>&1
if errorlevel 1 (
  echo Git is not installed. Install it from https://git-scm.com/download/win and run this again.
  pause
  exit /b 1
)
if not exist ".git" (
  git init -b main || goto :fail
)
git config user.name >nul 2>&1 || (
  set /p GITNAME=Your name for commits:
  set /p GITMAIL=Your GitHub email:
)
if defined GITNAME git config user.name "%GITNAME%"
if defined GITMAIL git config user.email "%GITMAIL%"
git add -A || goto :fail
git commit -m "SkillPulse AI: syllabus vs industry skill-gap analysis with ML models" || echo (nothing new to commit)
git remote get-url origin >nul 2>&1 || git remote add origin %REPO%
git branch -M main
git push -u origin main || goto :fail
echo.
echo Done: %REPO:.git=%
echo Add your teammates: GitHub repo - Settings - Collaborators - Add people.
pause
exit /b 0
:fail
echo.
echo Push failed. Check that the empty repository exists at %REPO:.git=% and that you are signed in.
pause
exit /b 1
