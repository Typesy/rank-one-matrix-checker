@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run Setup Checker.bat first.
  pause
  exit /b 1
)
set PYTHONUTF8=1
".venv\Scripts\python.exe" -m rankone.gui
if errorlevel 1 (
  echo The checker closed with an error. Please save the message above.
  pause
  exit /b 1
)
