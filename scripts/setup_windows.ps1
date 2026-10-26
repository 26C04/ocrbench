<#
.SYNOPSIS
    Prepare a fresh Windows PC for phase 1 (docs/windows-setup.md §2–5).

.DESCRIPTION
    Run from the repository root in an ordinary (non-administrator) PowerShell
    after cloning the repository:

        powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1

    Installs uv and Ollama with winget if they are missing, syncs the locked
    environment, runs the test suite, creates the data and run folders, sets
    OCRBENCH_DATA_DIR and OCRBENCH_RUNS_DIR, and pulls the five approved models.
    It is safe to run again: finished steps are skipped and pulls resume.
#>
[CmdletBinding()]
param(
    [string]$BaseDir = "D:\OCRBench",
    [switch]$SkipModels
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
$models = @("qwen3.5:9b", "minicpm-v4.5:8b", "glm-ocr:bf16", "gemma4:12b", "ministral-3:14b")

function Write-Step([string]$Text) {
    Write-Host ""
    Write-Host "=== $Text ===" -ForegroundColor Cyan
}

function Update-Path {
    # winget installs update the user PATH, which this session has not seen yet.
    $machine = [Environment]::GetEnvironmentVariable("Path", "Machine")
    $user = [Environment]::GetEnvironmentVariable("Path", "User")
    $env:Path = "$machine;$user"
}

function Install-IfMissing([string]$Command, [string]$WingetId) {
    if (Get-Command $Command -ErrorAction SilentlyContinue) {
        Write-Host "$Command is already installed."
        return
    }
    Write-Host "Installing $WingetId ..."
    & winget install --id=$WingetId -e --accept-package-agreements --accept-source-agreements
    Update-Path
    if (-not (Get-Command $Command -ErrorAction SilentlyContinue)) {
        throw "$Command is still unavailable. If the install was blocked, ask IT for an approved installation; do not bypass Windows security."
    }
}

Write-Step "GPU and driver"
& nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

Write-Step "Install uv and Ollama"
Install-IfMissing "uv" "astral-sh.uv"
Install-IfMissing "ollama" "Ollama.Ollama"
& uv --version
& ollama --version

Write-Step "Python environment"
& uv sync
if ($LASTEXITCODE -ne 0) { throw "uv sync failed." }
& uv run ocrbench --version
& uv run pytest -q
if ($LASTEXITCODE -ne 0) { throw "The test suite failed; stop and report the output." }

Write-Step "Folders and environment variables"
$dataDir = Join-Path $BaseDir "data"
$runsDir = Join-Path $BaseDir "runs"
foreach ($dir in @((Join-Path $dataDir "raw"), (Join-Path $dataDir "gt"), $runsDir, (Join-Path $BaseDir "samples"))) {
    New-Item -ItemType Directory -Force $dir | Out-Null
    Write-Host "Ready: $dir"
}
& setx OCRBENCH_DATA_DIR $dataDir | Out-Null
& setx OCRBENCH_RUNS_DIR $runsDir | Out-Null
$env:OCRBENCH_DATA_DIR = $dataDir
$env:OCRBENCH_RUNS_DIR = $runsDir
Write-Host "OCRBENCH_DATA_DIR = $dataDir"
Write-Host "OCRBENCH_RUNS_DIR = $runsDir"

if (-not $SkipModels) {
    Write-Step "Wait for the Ollama server"
    $ready = $false
    for ($i = 0; $i -lt 30 -and -not $ready; $i++) {
        try {
            Invoke-RestMethod -Uri "http://localhost:11434/api/version" -TimeoutSec 2 | Out-Null
            $ready = $true
        } catch {
            if ($i -eq 0) { Write-Host "Start Ollama from the Start menu if it is not running ..." }
            Start-Sleep -Seconds 2
        }
    }
    if (-not $ready) { throw "Ollama is not running. Start it from the Start menu, then run this script again." }

    Write-Step "Pull the five approved models"
    foreach ($model in $models) {
        Write-Host "Pulling $model ..."
        & ollama pull $model
        if ($LASTEXITCODE -ne 0) { throw "Pull failed for $model. Run this script again to resume." }
    }
    & ollama list
    Write-Host "Copy each model's ID (digest) from the list above into the benchmark notes."
}

Write-Step "Done"
Write-Host "Next: open scripts\gt_entry.html in Edge, choose $dataDir, and enter the ground truth."
