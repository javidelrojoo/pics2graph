@echo off
rem pics2graph launcher: double-click to open the graphical interface.
rem You can also drag an image onto this file to open it directly.
title pics2graph
cd /d "%~dp0"

rem Use the "py" launcher if available, otherwise "python".
where py >nul 2>nul
if %errorlevel%==0 (set "PY=py") else (set "PY=python")

%PY% gui.py %*

if errorlevel 1 (
    echo.
    echo pics2graph closed with an error - see the messages above.
    pause
)
