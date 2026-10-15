$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) { throw "Run INSTALL.bat first." }
& $python download_models.py
Write-Host "Models downloaded. Now run START.bat." -ForegroundColor Green
Read-Host "Press ENTER to close"
