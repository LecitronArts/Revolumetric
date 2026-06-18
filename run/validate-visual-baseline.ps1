param(
    [string]$Manifest = (Join-Path $PSScriptRoot "visual-baselines.json"),
    [string]$OutputDir = "target\visual-baseline",
    [switch]$Nrd,
    [string]$NrdRoot = (Join-Path $PSScriptRoot "nrd")
)

$ErrorActionPreference = "Stop"

function Assert-CaptureMetadata {
    param(
        [Parameter(Mandatory = $true)]
        [object]$Metadata,
        [Parameter(Mandatory = $true)]
        [object]$Case,
        [Parameter(Mandatory = $true)]
        [int]$CaptureFrame
    )

    if ([int64]$Metadata.frame_index -ne $CaptureFrame) {
        throw "capture metadata frame_index was $($Metadata.frame_index), expected $CaptureFrame."
    }
    if ($Metadata.vpt_debug_view -ne $Case.debugView) {
        throw "capture metadata vpt_debug_view was $($Metadata.vpt_debug_view), expected $($Case.debugView)."
    }
    if ($Metadata.denoiser_mode -ne $Case.denoiser) {
        throw "capture metadata denoiser_mode was $($Metadata.denoiser_mode), expected $($Case.denoiser)."
    }
    if ($Metadata.effective_denoiser_mode -ne $Case.expectedEffectiveDenoiser) {
        throw "capture metadata effective_denoiser_mode was $($Metadata.effective_denoiser_mode), expected $($Case.expectedEffectiveDenoiser)."
    }
    if ([int]$Metadata.width -le 0 -or [int]$Metadata.height -le 0) {
        throw "capture metadata dimensions must be positive."
    }
}

function Read-PpmHeader {
    param(
        [Parameter(Mandatory = $true)]
        [byte[]]$Bytes
    )

    $lineEnds = New-Object System.Collections.Generic.List[int]
    for ($i = 0; $i -lt $Bytes.Length -and $lineEnds.Count -lt 3; $i++) {
        if ($Bytes[$i] -eq 10) {
            $lineEnds.Add($i)
        }
    }
    if ($lineEnds.Count -lt 3) {
        throw "PPM header is incomplete."
    }

    $encoding = [System.Text.Encoding]::ASCII
    $magic = $encoding.GetString($Bytes, 0, $lineEnds[0])
    $dimsStart = $lineEnds[0] + 1
    $dimsLength = $lineEnds[1] - $dimsStart
    $dims = $encoding.GetString($Bytes, $dimsStart, $dimsLength).Trim().Split(" ", [System.StringSplitOptions]::RemoveEmptyEntries)
    $maxStart = $lineEnds[1] + 1
    $maxLength = $lineEnds[2] - $maxStart
    $maxValue = $encoding.GetString($Bytes, $maxStart, $maxLength)

    if ($magic -ne "P6") {
        throw "PPM magic was $magic, expected P6."
    }
    if ($dims.Count -ne 2) {
        throw "PPM dimensions line is invalid."
    }
    if ($maxValue -ne "255") {
        throw "PPM max value was $maxValue, expected 255."
    }

    [PSCustomObject]@{
        Width = [int]$dims[0]
        Height = [int]$dims[1]
        DataOffset = $lineEnds[2] + 1
    }
}

function Assert-PpmMatchesMetadata {
    param(
        [Parameter(Mandatory = $true)]
        [string]$PpmPath,
        [Parameter(Mandatory = $true)]
        [object]$Metadata
    )

    $bytes = [System.IO.File]::ReadAllBytes($PpmPath)
    $header = Read-PpmHeader -Bytes $bytes
    if ($header.Width -ne [int]$Metadata.width -or $header.Height -ne [int]$Metadata.height) {
        throw "PPM dimensions $($header.Width)x$($header.Height) did not match metadata $($Metadata.width)x$($Metadata.height)."
    }
    $expectedBytes = $header.DataOffset + ($header.Width * $header.Height * 3)
    if ($bytes.Length -ne $expectedBytes) {
        throw "PPM byte length was $($bytes.Length), expected $expectedBytes."
    }
}

function Assert-PpmHasNonZeroRgb {
    param(
        [Parameter(Mandatory = $true)]
        [string]$PpmPath
    )

    $bytes = [System.IO.File]::ReadAllBytes($PpmPath)
    $header = Read-PpmHeader -Bytes $bytes
    for ($i = $header.DataOffset; $i -lt $bytes.Length; $i++) {
        if ($bytes[$i] -ne 0) {
            return
        }
    }
    throw "PPM capture contains only zero RGB bytes: $PpmPath"
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$manifestPath = (Resolve-Path -LiteralPath $Manifest).Path
$manifestData = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
$captureFrame = [int]$manifestData.captureFrame
$frames = [int]$manifestData.frames
if ($captureFrame -lt 0 -or $frames -le $captureFrame) {
    throw "visual baseline manifest requires frames > captureFrame >= 0."
}

$previousLocation = Get-Location
$previousEnv = @{}
foreach ($name in @(
    "REVOLUMETRIC_SHADER_COMPILE",
    "REVOLUMETRIC_CAPTURE_FRAME",
    "REVOLUMETRIC_CAPTURE_DIR",
    "REVOLUMETRIC_CAPTURE_PREFIX",
    "REVOLUMETRIC_DENOISER",
    "REVOLUMETRIC_VPT_DEBUG_VIEW",
    "REVOLUMETRIC_EXIT_AFTER_FRAMES",
    "REVOLUMETRIC_NRD_ROOT",
    "PATH"
)) {
    $previousEnv[$name] = [Environment]::GetEnvironmentVariable($name, "Process")
}

try {
    Set-Location $repoRoot
    New-Item -ItemType Directory -Force $OutputDir | Out-Null

    foreach ($case in $manifestData.cases) {
        if ($case.requiresNrd -and -not $Nrd) {
            Write-Host "skip visual baseline $($case.name): requires -Nrd"
            continue
        }

        $caseOutputDir = Join-Path $OutputDir $case.name
        Remove-Item -LiteralPath $caseOutputDir -Recurse -Force -ErrorAction SilentlyContinue
        New-Item -ItemType Directory -Force $caseOutputDir | Out-Null

        $env:REVOLUMETRIC_CAPTURE_FRAME = "$captureFrame"
        $env:REVOLUMETRIC_CAPTURE_DIR = $caseOutputDir
        $env:REVOLUMETRIC_CAPTURE_PREFIX = $case.name
        $env:REVOLUMETRIC_EXIT_AFTER_FRAMES = "$frames"
        $env:REVOLUMETRIC_DENOISER = $case.denoiser
        if ($case.debugView -eq "final") {
            Remove-Item Env:\REVOLUMETRIC_VPT_DEBUG_VIEW -ErrorAction SilentlyContinue
        } else {
            $env:REVOLUMETRIC_VPT_DEBUG_VIEW = $case.debugView
        }

        Write-Host "==> visual baseline $($case.name)"
        if ($case.requiresNrd) {
            & .\run\validate-nrd.ps1 -NrdRoot $NrdRoot -Denoiser $case.denoiser -DebugView $case.debugView -Frames $frames
        } else {
            $env:REVOLUMETRIC_SHADER_COMPILE = "strict"
            cargo run --features desktop --bin revolumetric
        }
        if ($LASTEXITCODE -ne 0) {
            throw "visual baseline run failed for $($case.name) with exit code $LASTEXITCODE."
        }

        $stem = "{0}_{1:D6}" -f $case.name, $captureFrame
        $jsonPath = Join-Path $caseOutputDir "$stem.json"
        $ppmPath = Join-Path $caseOutputDir "$stem.ppm"
        if (-not (Test-Path -LiteralPath $jsonPath)) {
            throw "capture metadata was not written: $jsonPath"
        }
        if (-not (Test-Path -LiteralPath $ppmPath)) {
            throw "capture PPM was not written: $ppmPath"
        }

        $metadata = Get-Content -LiteralPath $jsonPath -Raw | ConvertFrom-Json
        Assert-CaptureMetadata -Metadata $metadata -Case $case -CaptureFrame $captureFrame
        Assert-PpmMatchesMetadata -PpmPath $ppmPath -Metadata $metadata
        Assert-PpmHasNonZeroRgb -PpmPath $ppmPath
    }
} finally {
    Set-Location $previousLocation
    foreach ($entry in $previousEnv.GetEnumerator()) {
        [Environment]::SetEnvironmentVariable($entry.Key, $entry.Value, "Process")
    }
}
