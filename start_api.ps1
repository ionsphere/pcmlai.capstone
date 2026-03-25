# Start API Server - PowerShell Script

$ErrorActionPreference = "Stop"

# Colors
function Write-Header { Write-Host $args[0] -ForegroundColor Cyan }
function Write-Success { Write-Host $args[0] -ForegroundColor Green }
function Write-Error { Write-Host $args[0] -ForegroundColor Red }
function Write-Warning { Write-Host $args[0] -ForegroundColor Yellow }
function Write-Info { Write-Host $args[0] -ForegroundColor White }

Write-Header "================================================================================"
Write-Header "Clothing Price Predictor API"
Write-Header "================================================================================"
Write-Host ""

# Check Python installation
try {
    $pythonVersion = python --version 2>&1
    Write-Success "Python found: $pythonVersion"
} catch {
    Write-Error "Error: Python is not installed or not in PATH"
    Write-Error "Please install Python 3.9+ from https://www.python.org/"
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Info "Select an option:"
Write-Host ""
Write-Host "  1. Start API (development mode with auto-reload)"
Write-Host "  2. Start API (production mode, 4 workers)"
Write-Host "  3. Test API connection (quick health check)"
Write-Host "  4. View API documentation (browser)"
Write-Host "  5. Custom configuration"
Write-Host "  6. Exit"
Write-Host ""

$choice = Read-Host "Enter your choice (1-6)"

switch ($choice) {
    "1" {
        Write-Header "`nStarting API in development mode..."
        Write-Info "API will be available at http://localhost:8000"
        Write-Info "Documentation at http://localhost:8000/docs"
        Write-Host ""
        python scripts\start_api.py --host 0.0.0.0 --port 8000 --reload --log-level info
    }
    "2" {
        Write-Header "`nStarting API in production mode..."
        Write-Info "API will be available at http://localhost:8000"
        Write-Host ""
        python scripts\start_api.py --host 0.0.0.0 --port 8000 --workers 4 --log-level warning
    }
    "3" {
        Write-Header "`nTesting API connection..."
        Start-Sleep -Seconds 1
        try {
            $response = Invoke-WebRequest -Uri "http://localhost:8000/health" -UseBasicParsing
            Write-Success "API is running!"
            Write-Host "Status: $($response.StatusCode)"
            Write-Host "Response: $($response.Content)"
        } catch {
            Write-Error "API is not running or not reachable"
            Write-Host $_.Exception.Message
        }
        Read-Host "`nPress Enter to continue"
    }
    "4" {
        Write-Header "`nOpening API documentation in browser..."
        Start-Process "http://localhost:8000/docs"
        Read-Host "Press Enter to continue"
    }
    "5" {
        Write-Header "`nCustom configuration mode"
        Write-Host ""
        
        $host_input = Read-Host "Host [0.0.0.0]"
        $host = if ([string]::IsNullOrWhiteSpace($host_input)) { "0.0.0.0" } else { $host_input }
        
        $port_input = Read-Host "Port [8000]"
        $port = if ([string]::IsNullOrWhiteSpace($port_input)) { "8000" } else { $port_input }
        
        $reload_input = Read-Host "Enable auto-reload? (y/n) [y]"
        $reload = if ([string]::IsNullOrWhiteSpace($reload_input)) { "y" } else { $reload_input }
        
        $workers_input = Read-Host "Number of workers [1]"
        $workers = if ([string]::IsNullOrWhiteSpace($workers_input)) { "1" } else { $workers_input }
        
        $log_level_input = Read-Host "Log level (debug/info/warning/error) [info]"
        $log_level = if ([string]::IsNullOrWhiteSpace($log_level_input)) { "info" } else { $log_level_input }
        
        $reload_flag = if ($reload -eq "y") { "--reload" } else { "" }
        
        Write-Header "`nStarting API with custom configuration..."
        python scripts\start_api.py --host $host --port $port $reload_flag --workers $workers --log-level $log_level
    }
    "6" {
        Write-Success "`nExiting..."
        exit 0
    }
    default {
        Write-Error "`nInvalid choice. Please try again."
    }
}

Write-Host ""
Write-Header "================================================================================"
Write-Header "Done!"
Write-Header "================================================================================"
Read-Host "Press Enter to exit"
