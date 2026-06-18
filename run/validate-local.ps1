param(
    [switch]$StrictShaders,
    [switch]$Nrd,
    [switch]$NrdRuntime,
    [string]$NrdRoot = (Join-Path $PSScriptRoot "nrd"),
    [int]$NrdFrames = 3
)

$ErrorActionPreference = "Stop"

if ($NrdFrames -lt 0) {
    throw "NrdFrames must be >= 0."
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$previousLocation = Get-Location
$previousEnv = @{}
foreach ($name in @(
    "REVOLUMETRIC_SHADER_COMPILE",
    "REVOLUMETRIC_NRD_ROOT"
)) {
    $previousEnv[$name] = [Environment]::GetEnvironmentVariable($name, "Process")
}

function Invoke-CheckedCommand {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Label,
        [Parameter(Mandatory = $true)]
        [scriptblock]$Command
    )

    Write-Host "==> $Label"
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "$Label failed with exit code $LASTEXITCODE."
    }
}

try {
    Set-Location $repoRoot

    Invoke-CheckedCommand "cargo fmt --check" { cargo fmt --check }

    $env:REVOLUMETRIC_SHADER_COMPILE = "skip"
    Invoke-CheckedCommand "REVOLUMETRIC_SHADER_COMPILE=skip cargo test --lib" {
        cargo test --lib
    }
    Invoke-CheckedCommand "REVOLUMETRIC_SHADER_COMPILE=skip cargo clippy --all-targets -- -D warnings" {
        cargo clippy --all-targets -- -D warnings
    }

    if ($StrictShaders) {
        $env:REVOLUMETRIC_SHADER_COMPILE = "strict"
        Invoke-CheckedCommand "REVOLUMETRIC_SHADER_COMPILE=strict cargo test --lib" {
            cargo test --lib
        }
        Invoke-CheckedCommand "REVOLUMETRIC_SHADER_COMPILE=strict cargo build --lib" {
            cargo build --lib
        }
    }

    if ($Nrd) {
        $resolvedNrdRoot = (Resolve-Path -LiteralPath $NrdRoot -ErrorAction Stop).Path
        $env:REVOLUMETRIC_NRD_ROOT = $resolvedNrdRoot
        if (-not $StrictShaders) {
            $env:REVOLUMETRIC_SHADER_COMPILE = "strict"
        }
        Invoke-CheckedCommand "REVOLUMETRIC_SHADER_COMPILE=$env:REVOLUMETRIC_SHADER_COMPILE cargo test --lib --features nrd" {
            cargo test --lib --features nrd
        }
    }

    if ($NrdRuntime) {
        & .\run\validate-nrd.ps1 -NrdRoot $NrdRoot -Denoiser reblur -Frames $NrdFrames
        if ($LASTEXITCODE -ne 0) {
            throw ".\run\validate-nrd.ps1 -Denoiser reblur -Frames $NrdFrames failed with exit code $LASTEXITCODE."
        }
    }
} finally {
    Set-Location $previousLocation
    foreach ($entry in $previousEnv.GetEnumerator()) {
        [Environment]::SetEnvironmentVariable($entry.Key, $entry.Value, "Process")
    }
}
