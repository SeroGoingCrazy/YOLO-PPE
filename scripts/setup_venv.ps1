param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

Write-Host "Creating virtual environment in .venv ..."
& $Python -m venv .venv

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    throw "Virtual environment was not created successfully."
}

Write-Host "Installing dependencies ..."
& ".venv\Scripts\python.exe" -m pip install --upgrade pip
& ".venv\Scripts\python.exe" -m pip install -r requirements.txt

Write-Host "Done. Activate with: .\.venv\Scripts\Activate.ps1"

