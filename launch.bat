@echo off
rem ============================================================================
rem  Photogrammetry Harness - double-click launcher
rem
rem  Runs the harness as a SINGLE process: FastAPI serves both the API and the
rem  built frontend, so there is one port, one window, and one thing to shut down.
rem
rem  Shutdown: uvicorn runs in the FOREGROUND of this window on purpose. Closing
rem  the window terminates it, and any tool it launched - ffmpeg, COLMAP, OpenMVS -
rem  dies with it, because those children are held in a Win32 Job Object created
rem  with KILL_ON_JOB_CLOSE and so cannot outlive the parent holding the GPU.
rem ============================================================================

setlocal
cd /d "%~dp0"

set "PORT=8756"
set "HOST=127.0.0.1"
set "PYTHON=.venv\Scripts\python.exe"
set "URL=http://%HOST%:%PORT%/"

title Photogrammetry Harness

echo.
echo   Photogrammetry Harness
echo   ----------------------
echo.

rem --- the virtual environment must exist -----------------------------------
if not exist "%PYTHON%" goto :no_venv

rem --- rebuild the frontend only when it is actually out of date -------------
rem The comparison lives in scripts\needs-build.ps1. Expressing it inline would
rem mean escaping a PowerShell pipeline through both cmd and `for /f`, which is
rem fiddly enough to get silently wrong - cmd hands PowerShell a literal ^| and
rem the check errors out instead of answering.
powershell -NoProfile -ExecutionPolicy Bypass -File "scripts\needs-build.ps1"
if not errorlevel 1 goto :check_port

echo   Building the interface...
where npm >nul 2>&1
if errorlevel 1 goto :no_npm
pushd web
if not exist "node_modules" call npm install --no-fund --no-audit
call npm run build
if errorlevel 1 goto :build_failed
popd
echo   Built.
echo.

rem --- refuse to fight an instance that is already running -------------------
:check_port
powershell -NoProfile -Command "if (Get-NetTCPConnection -LocalPort %PORT% -State Listen -ErrorAction SilentlyContinue) { exit 1 } else { exit 0 }"
if errorlevel 1 goto :already_running

rem --- open the browser once the server actually answers ---------------------
rem Detached so it can poll while uvicorn starts up in the foreground. It waits
rem for a real 200 from the health endpoint rather than guessing a fixed delay,
rem so the browser never lands on a connection-refused page.
start "" /b powershell -NoProfile -WindowStyle Hidden -Command "for ($i=0; $i -lt 120; $i++) { try { $null = Invoke-WebRequest -Uri 'http://%HOST%:%PORT%/api/health' -UseBasicParsing -TimeoutSec 1; Start-Process '%URL%'; break } catch { Start-Sleep -Milliseconds 500 } }"

echo   Starting on %URL%
echo.
echo   Close this window to shut the harness down.
echo   ----------------------------------------------------------------
echo.

"%PYTHON%" -m uvicorn pgh.main:app --host %HOST% --port %PORT% --app-dir server

echo.
echo   Harness stopped.
timeout /t 3 >nul
goto :eof

rem --- failure paths ---------------------------------------------------------

:no_venv
echo   [X] No Python environment found at %PYTHON%
echo.
echo   Create it first, from this folder:
echo       py -3.12 -m venv .venv
echo       .venv\Scripts\python.exe -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu130
echo       .venv\Scripts\python.exe -m pip install -e ./sam2 --no-build-isolation
echo.
pause
exit /b 1

:no_npm
echo   [X] npm is not on PATH, so the interface cannot be built.
echo       Install Node.js, or build web\dist elsewhere and copy it here.
echo.
pause
exit /b 1

:build_failed
popd
echo.
echo   [X] The interface failed to build. See the messages above.
echo.
pause
exit /b 1

:already_running
echo   [!] Something is already listening on port %PORT%.
echo       The harness is probably already open - reopening %URL%
start "" "%URL%"
echo.
timeout /t 5 >nul
exit /b 0
