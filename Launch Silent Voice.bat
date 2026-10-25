@echo off
rem Double-click to start the whole Silent Voice demo (demo/launch.py). Options go after it, e.g. from a terminal:
rem   "Launch Silent Voice.bat" --no-emg        pads off
rem   "Launch Silent Voice.bat" --check         pre-flight checks only
rem   "Launch Silent Voice.bat" --rehearse      no camera or pads: session 3 plays when a call connects
setlocal
cd /d "%~dp0"
set "PY="
if defined RECLAIM_WORKDIR if exist "%RECLAIM_WORKDIR%\venv-call\Scripts\python.exe" set "PY=%RECLAIM_WORKDIR%\venv-call\Scripts\python.exe"
if not defined PY if exist "%~dp0..\venv-call\Scripts\python.exe" set "PY=%~dp0..\venv-call\Scripts\python.exe"
if not defined PY if exist "%~dp0..\..\venv-call\Scripts\python.exe" set "PY=%~dp0..\..\venv-call\Scripts\python.exe"
if not defined PY (
    echo Could not find venv-call. Set RECLAIM_WORKDIR to the folder that holds venv-call ^(see DEMO.md^).
    pause
    exit /b 1
)
"%PY%" -m demo.launch %*
echo.
pause
