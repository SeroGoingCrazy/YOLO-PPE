param(
    [Parameter(Mandatory = $true)]
    [string]$Dataset,
    [string]$OutDir = "datasets\raw"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    throw "Please create venv first: powershell -ExecutionPolicy Bypass -File scripts\setup_venv.ps1"
}

if (-not (Test-Path $OutDir)) {
    New-Item -ItemType Directory -Path $OutDir | Out-Null
}

Write-Host "Installing Kaggle API in venv ..."
& ".venv\Scripts\python.exe" -m pip install kaggle

Write-Host "Downloading dataset: $Dataset"
$kaggleExe = ".venv\Scripts\kaggle.exe"
if (Test-Path $kaggleExe) {
    & $kaggleExe datasets download -d $Dataset -p $OutDir --unzip
} else {
    # Fallback for environments without kaggle.exe wrapper.
    & ".venv\Scripts\python.exe" -m kaggle.cli datasets download -d $Dataset -p $OutDir --unzip
}

Write-Host "Done. Please normalize to datasets\ppe-construction and verify data\ppe_kaggle.yaml path."

