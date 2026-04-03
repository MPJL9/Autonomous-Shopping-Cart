@echo off
setlocal
cd /d "%~dp0.."
python -c "import fastapi, httpx, pydantic, uvicorn" >nul 2>&1
if errorlevel 1 (
  echo Installing dashboard dependencies...
  python -m pip install -e .[dev]
)

start "" powershell -NoProfile -WindowStyle Hidden -Command ^
  "$deadline=(Get-Date).AddSeconds(45); while((Get-Date)-lt $deadline){ try { Invoke-WebRequest 'http://127.0.0.1:8000/api/status' -UseBasicParsing -TimeoutSec 2 | Out-Null; Start-Process 'http://127.0.0.1:8000'; exit 0 } catch { Start-Sleep -Milliseconds 500 } }"

echo Starting Veda dashboard on http://127.0.0.1:8000
python -m cart_stack.dashboard.run
