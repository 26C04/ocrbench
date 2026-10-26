<#
.SYNOPSIS
    Run ADR-0001 phase one unattended: smoke each approved model, run all five
    on the Development split, write reports, and rank the models (ADR §8-1/§8-2).

.DESCRIPTION
    Run from the repository root in an ordinary (non-administrator) PowerShell:

        powershell -ExecutionPolicy Bypass -File scripts\phase1.ps1 -SampleImage D:\OCRBench\samples\anonymous.png

    Prerequisites (docs/windows-setup.md): Ollama is running, all five approved
    models are pulled, OCRBENCH_DATA_DIR holds a validated Development dataset
    with ground truth, and OCRBENCH_RUNS_DIR exists.

    A failure for one model is logged and the script moves on to the next one.
    Everything printed is also saved to a transcript in OCRBENCH_RUNS_DIR.
#>
[CmdletBinding()]
param(
    # An anonymous sample image for the smoke step. Omit with -SkipSmoke.
    [string]$SampleImage,
    [switch]$SkipSmoke,
    [string[]]$Models = @(
        "qwen3.5:9b",
        "minicpm-v4.5:8b",
        "glm-ocr:bf16",
        "gemma4:12b",
        "ministral-3:14b"
    )
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

function Use-UserVariable([string]$Name) {
    # setx only affects new terminals; read the user value if this one predates it.
    $value = [Environment]::GetEnvironmentVariable($Name, "Process")
    if (-not $value) { $value = [Environment]::GetEnvironmentVariable($Name, "User") }
    if (-not $value) { throw "$Name is not set. See docs/windows-setup.md section 5." }
    [Environment]::SetEnvironmentVariable($Name, $value, "Process")
    return $value
}

function Invoke-Ocrbench([string[]]$Arguments) {
    # Returns the parsed --json payload; the exit code is left in $LASTEXITCODE.
    $lines = & uv run ocrbench @Arguments --json
    $json = $lines | Where-Object { $_ -like "{*" } | Select-Object -Last 1
    if (-not $json) { return $null }
    $payload = $json | ConvertFrom-Json
    if ($payload.status -eq "error") { Write-Host "ocrbench error: $($payload.error)" -ForegroundColor Red }
    return $payload
}

function Write-Step([string]$Text) {
    Write-Host ""
    Write-Host "=== $Text ===" -ForegroundColor Cyan
}

$dataDir = Use-UserVariable "OCRBENCH_DATA_DIR"
$runsDir = Use-UserVariable "OCRBENCH_RUNS_DIR"
$stamp = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssZ")
$transcript = Join-Path $runsDir "phase1-$stamp.log"
Start-Transcript -Path $transcript | Out-Null

try {
    Write-Step "Preflight"
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) { throw "uv is not installed." }
    if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) { throw "ollama is not installed." }
    try {
        $version = Invoke-RestMethod -Uri "http://localhost:11434/api/version" -TimeoutSec 5
    } catch {
        throw "Ollama is not running. Start it from the Start menu and run this script again."
    }
    Write-Host "Ollama server version: $($version.version)"
    $pulled = (& ollama list) -join "`n"
    $missing = @($Models | Where-Object { $pulled -notmatch [regex]::Escape($_) })
    if ($missing.Count -gt 0) { throw "Models not pulled yet: $($missing -join ', ')" }
    if (-not $SkipSmoke) {
        if (-not $SampleImage) { throw "Pass -SampleImage <anonymous image>, or -SkipSmoke." }
        if (-not (Test-Path -LiteralPath $SampleImage -PathType Leaf)) {
            throw "Sample image not found: $SampleImage"
        }
    }

    Write-Step "Environment record"
    & nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
    & ollama list

    Write-Step "Dataset"
    $validation = Invoke-Ocrbench @("dataset", "validate", "--root", $dataDir)
    if ($LASTEXITCODE -ne 0) { throw "Dataset validation failed; fix the errors above first." }
    Write-Host "Documents in manifest: $($validation.n_docs)"
    foreach ($warning in $validation.split_count_warnings) { Write-Host "Warning: $warning" }
    $fingerprint = Invoke-Ocrbench @("dataset", "fingerprint", "--root", $dataDir)
    Write-Host "Dataset fingerprint: $($fingerprint.fingerprint)"

    $runDirs = @()
    $summary = @()
    foreach ($model in $Models) {
        Write-Step "Model $model"
        $smokeOk = "skipped"
        if (-not $SkipSmoke) {
            $smoke = Invoke-Ocrbench @("smoke", "--model", $model, "--image", $SampleImage)
            $smokeOk = if ($LASTEXITCODE -eq 0) { "ok" } else { "FAILED" }
            Write-Host "Smoke: $smokeOk"
            & ollama ps
        }

        $runId = "p1-" + ($model -replace "[^A-Za-z0-9]", "-") + "-" + $stamp
        $started = Get-Date
        $run = Invoke-Ocrbench @("run", "--model", $model, "--split", "dev", "--prompt-name", "base", "--run-id", $runId)
        $minutes = [math]::Round(((Get-Date) - $started).TotalMinutes, 1)
        if ($LASTEXITCODE -ne 0 -or -not $run.run_dir) {
            Write-Host "Run FAILED for $model after $minutes min; continuing." -ForegroundColor Red
            $summary += [pscustomobject]@{ Model = $model; Smoke = $smokeOk; Run = "FAILED"; Minutes = $minutes }
        } else {
            Write-Host "Run finished in $minutes min: $($run.run_dir)"
            Invoke-Ocrbench @("report", "--run-dir", $run.run_dir) | Out-Null
            $runDirs += $run.run_dir
            $summary += [pscustomobject]@{ Model = $model; Smoke = $smokeOk; Run = "ok"; Minutes = $minutes }
        }
        # Free VRAM so the next model is measured without leftovers.
        try { & ollama stop $model | Out-Null } catch { }
    }

    Write-Step "Summary"
    $summary | Format-Table -AutoSize | Out-String | Write-Host

    if ($runDirs.Count -gt 0) {
        Write-Step "Ranking"
        $rankingFile = Join-Path $runsDir "phase1-ranking-$stamp.md"
        & uv run python -m scripts.rank_models --run-dirs @runDirs --out $rankingFile
        Write-Host ""
        Write-Host "Ranking saved to $rankingFile" -ForegroundColor Green
    } else {
        Write-Host "No run finished, so there is nothing to rank." -ForegroundColor Red
    }
} finally {
    Stop-Transcript | Out-Null
    Write-Host "Full log: $transcript"
}
