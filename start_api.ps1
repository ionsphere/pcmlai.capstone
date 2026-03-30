$ErrorActionPreference = "Stop"

Write-Host "Clothing Price Predictor API"

try {
    $pythonVersion = python --version 2>&1
    Write-Host "Python found: $pythonVersion"
} catch {
    Write-Host "Error: Python is not installed or not in PATH"
    Write-Host "Please install Python 3.9+ from https://www.python.org/"
    Read-Host "Press Enter to exit"
    exit 1
}

python scripts\start_api.py --host 0.0.0.0 --port 8000 --reload --log-level info
Write-Host "Done!"
Read-Host "Press Enter to exit"
