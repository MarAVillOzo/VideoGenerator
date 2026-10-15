$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) { throw "Run INSTALL.bat first." }
if (-not (Test-Path ".\models\ltxv-2b-0.9.8-distilled.safetensors")) { throw "Run DOWNLOAD_MODELS.bat first." }
& $python app.py
