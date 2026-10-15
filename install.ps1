$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host " LOCAL VIDEO STUDIO - RTX 3060 12 GB - INSTALLER V2" -ForegroundColor Cyan
Write-Host "=====================================================" -ForegroundColor Cyan

function Find-Python {
    $tests = @(
        @{Cmd="py"; Arg="-3.11"},
        @{Cmd="py"; Arg="-3.10"},
        @{Cmd="python"; Arg=""}
    )
    foreach ($t in $tests) {
        try {
            if ($t.Arg -eq "") {
                $ver = & $t.Cmd -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
            } else {
                $ver = & $t.Cmd $t.Arg -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
            }
            if ($ver -eq "3.10" -or $ver -eq "3.11") { return $t }
        } catch {}
    }
    return $null
}

$py = Find-Python
if ($null -eq $py) {
    Write-Host "Python 3.10/3.11 not found. Installing Python 3.11..." -ForegroundColor Yellow
    if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
        throw "Install Python 3.11 x64 manually, then run INSTALL.bat again."
    }
    winget install --id Python.Python.3.11 -e --accept-package-agreements --accept-source-agreements
    $py = @{Cmd="py"; Arg="-3.11"}
}

if (-not (Test-Path ".venv")) {
    if ($py.Arg -eq "") { & $py.Cmd -m venv .venv }
    else { & $py.Cmd $py.Arg -m venv .venv }
}

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$pip = Join-Path $PSScriptRoot ".venv\Scripts\pip.exe"

& $python -m pip install --upgrade pip setuptools wheel
& $pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
& $pip install -r requirements.txt

$vendor = Join-Path $PSScriptRoot "vendor"
$ltx = Join-Path $vendor "LTX-Video"

if (-not (Test-Path (Join-Path $ltx "pyproject.toml"))) {
    New-Item -ItemType Directory -Force -Path $vendor | Out-Null
    $zip = Join-Path $vendor "ltx-video-main.zip"
    $extract = Join-Path $vendor "_extract"
    if (Test-Path $extract) { Remove-Item $extract -Recurse -Force }

    Write-Host "Downloading official Lightricks LTX-Video source..." -ForegroundColor Cyan
    Invoke-WebRequest -Uri "https://github.com/Lightricks/LTX-Video/archive/refs/heads/main.zip" -OutFile $zip
    Expand-Archive -Path $zip -DestinationPath $extract -Force
    $found = Get-ChildItem $extract -Directory | Select-Object -First 1
    Move-Item $found.FullName $ltx
    Remove-Item $extract -Recurse -Force
    Remove-Item $zip -Force
}

Push-Location $ltx
& $pip install -e ".[inference-script]"
Pop-Location

& $python system_check.py

Write-Host ""
Write-Host "INSTALLATION COMPLETE." -ForegroundColor Green
Write-Host "Next: run DOWNLOAD_MODELS.bat once, then START.bat." -ForegroundColor Yellow
