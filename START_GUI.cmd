@echo off
setlocal
cd /d "%~dp0"
if defined SCREENQC_PYTHON (
    set "PY=%SCREENQC_PYTHON%"
) else if exist "D:\AI\miniForge1\python.exe" (
    set "PY=D:\AI\miniForge1\python.exe"
) else (
    set "PY=python"
)
"%PY%" -u "%~dp0main.py" gui
if errorlevel 1 (
    echo.
    echo ScreenQC could not start. Please check Python and installed dependencies.
    pause
)
