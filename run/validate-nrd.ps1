param(
    [string]$NrdRoot = (Join-Path $PSScriptRoot "nrd"),
    [ValidateSet("auto", "strict", "skip")]
    [string]$ShaderCompile = "strict",
    [ValidateSet("final", "nrd_normal_roughness", "nrd_viewz", "nrd_motion", "nrd_motion_z", "nrd_validation")]
    [string]$DebugView = "final",
    [ValidateSet("relax", "svgf", "off", "reblur")]
    [string]$Denoiser = "relax",
    [ValidateRange(0, 5)]
    [int]$AtrousIterations = 4,
    [int]$Frames = 3,
    [switch]$BuildOnly
)

$ErrorActionPreference = "Stop"

if ($Frames -lt 0) {
    throw "Frames must be >= 0. Use 0 for an interactive run without REVOLUMETRIC_EXIT_AFTER_FRAMES."
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$resolvedNrdRootInfo = Resolve-Path -LiteralPath $NrdRoot -ErrorAction SilentlyContinue
if ($null -eq $resolvedNrdRootInfo) {
    throw "NRD SDK root was not found: $NrdRoot. Put the SDK under run\nrd or pass -NrdRoot <path>."
}
$resolvedNrdRoot = $resolvedNrdRootInfo.Path

$requiredHeaders = @(
    "Include\NRD.h",
    "Include\NRDDescs.h",
    "Include\NRDSettings.h"
)
foreach ($relativePath in $requiredHeaders) {
    $fullPath = Join-Path $resolvedNrdRoot $relativePath
    if (-not (Test-Path -LiteralPath $fullPath)) {
        throw "NRD SDK root is missing $relativePath`: $resolvedNrdRoot"
    }
}

$libraryDirs = @(
    "_Bin",
    "Lib",
    "lib",
    "Build\Release",
    "build\Release",
    "build\lib"
) | ForEach-Object { Join-Path $resolvedNrdRoot $_ }

$libraryDir = $libraryDirs | Where-Object {
    (Test-Path -LiteralPath (Join-Path $_ "NRD.lib")) -or
    (Test-Path -LiteralPath (Join-Path $_ "libNRD.a"))
} | Select-Object -First 1

if ($null -eq $libraryDir) {
    throw "NRD SDK root has headers but no NRD.lib/libNRD.a under _Bin, Lib, lib, Build\Release, build\Release, or build\lib: $resolvedNrdRoot"
}

$previousEnv = @{}
foreach ($name in @(
    "REVOLUMETRIC_NRD_ROOT",
    "REVOLUMETRIC_SHADER_COMPILE",
    "REVOLUMETRIC_DENOISER",
    "REVOLUMETRIC_DENOISER_ATROUS_ITERATIONS",
    "REVOLUMETRIC_EXIT_AFTER_FRAMES",
    "REVOLUMETRIC_VPT_DEBUG_VIEW",
    "PATH"
)) {
    $previousEnv[$name] = [Environment]::GetEnvironmentVariable($name, "Process")
}
$previousLocation = Get-Location

try {
    Set-Location $repoRoot

    $env:REVOLUMETRIC_NRD_ROOT = $resolvedNrdRoot
    $env:REVOLUMETRIC_SHADER_COMPILE = $ShaderCompile
    $env:REVOLUMETRIC_DENOISER = $Denoiser
    $env:REVOLUMETRIC_DENOISER_ATROUS_ITERATIONS = "$AtrousIterations"
    $env:PATH = "$libraryDir;$env:PATH"

    if ($Frames -gt 0) {
        $env:REVOLUMETRIC_EXIT_AFTER_FRAMES = "$Frames"
    } else {
        Remove-Item Env:\REVOLUMETRIC_EXIT_AFTER_FRAMES -ErrorAction SilentlyContinue
    }

    if ($DebugView -eq "final") {
        Remove-Item Env:\REVOLUMETRIC_VPT_DEBUG_VIEW -ErrorAction SilentlyContinue
    } else {
        $env:REVOLUMETRIC_VPT_DEBUG_VIEW = $DebugView
    }

    Write-Host "NRD root: $resolvedNrdRoot"
    Write-Host "NRD library dir: $libraryDir"
    Write-Host "Shader compile: $ShaderCompile"
    Write-Host "Denoiser: $Denoiser"
    Write-Host "Frames: $Frames"
    Write-Host "Debug view: $DebugView"

    if ($BuildOnly) {
        cargo build --features "desktop nrd" --bin revolumetric
    } else {
        cargo run --features "desktop nrd" --bin revolumetric
    }

    if ($LASTEXITCODE -ne 0) {
        throw "cargo command failed with exit code $LASTEXITCODE."
    }
} finally {
    Set-Location $previousLocation
    foreach ($entry in $previousEnv.GetEnumerator()) {
        [Environment]::SetEnvironmentVariable($entry.Key, $entry.Value, "Process")
    }
}
