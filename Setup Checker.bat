@echo off
setlocal
cd /d "%~dp0"
echo Setting up Rank-one Checker...
py -3.12 -c "import tkinter" >nul 2>&1
if errorlevel 1 goto :missing
if not exist ".venv\Scripts\python.exe" py -3.12 -m venv .venv
if errorlevel 1 goto :failed
".venv\Scripts\python.exe" -m pip install .
if errorlevel 1 goto :failed
echo.
echo Ready. Double-click Open Checker.bat to start.
pause
exit /b 0
:missing
echo Install Python 3.12 for Windows from https://www.python.org/downloads/windows/
echo Keep the Python launcher and Tcl/Tk options selected, then run this file again.
pause
exit /b 1
:failed
echo Setup did not finish. Keep this window open to read the error above.
pause
exit /b 1
