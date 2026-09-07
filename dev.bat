@echo off
rem ============================================================================
rem  Development mode: Vite dev server + auto-reloading API, in two windows.
rem
rem  Use launch.bat for normal use. This one exists for editing the frontend:
rem  Vite serves it on 5173 with hot reload and proxies /api to the backend.
rem  Closing either window stops that half.
rem ============================================================================

setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo No Python environment. See launch.bat for setup.
    pause
    exit /b 1
)
if not exist "web\node_modules" pushd web ^& call npm install --no-fund --no-audit ^& popd

start "Harness API" cmd /k ".venv\Scripts\python.exe -m uvicorn pgh.main:app --port 8756 --app-dir server --reload --reload-dir server"
start "Harness UI"  cmd /k "cd web && npm run dev"

echo.
echo   API on http://127.0.0.1:8756   ^(auto-reloads on server\ changes^)
echo   UI  on http://127.0.0.1:5173   ^(hot reload, proxies /api^)
echo.
echo   Close each window to stop that half.
timeout /t 6 >nul
endlocal
