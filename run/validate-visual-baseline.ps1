param(
    [string]$Manifest = (Join-Path $PSScriptRoot "visual-baselines.json"),
    [string]$OutputDir = "target\visual-baseline",
    [switch]$Nrd,
    [switch]$Rt,
    [string]$NrdRoot = (Join-Path $PSScriptRoot "nrd")
)

$ErrorActionPreference = "Stop"

function Assert-MetadataBooleanField {
    param(
        [Parameter(Mandatory = $true)]
        [object]$Metadata,
        [Parameter(Mandatory = $true)]
        [string]$FieldName,
        [Parameter(Mandatory = $true)]
        [object]$ExpectedValue
    )

    if (-not ($Metadata.PSObject.Properties.Name -contains $FieldName)) {
        throw "capture metadata $FieldName was missing, expected $ExpectedValue."
    }

    $actualValue = $Metadata.PSObject.Properties[$FieldName].Value
    if ([bool]$actualValue -ne [bool]$ExpectedValue) {
        throw "capture metadata $FieldName was $actualValue, expected $ExpectedValue."
    }
}

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
    if ($Case.expectedRenderBackend -and $Metadata.render_backend -ne $Case.expectedRenderBackend) {
        throw "capture metadata render_backend was $($Metadata.render_backend), expected $($Case.expectedRenderBackend)."
    }
    if ($Case.renderMode -and $Metadata.render_mode -ne $Case.renderMode) {
        throw "capture metadata render_mode was $($Metadata.render_mode), expected $($Case.renderMode)."
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
    if ($Case.rtDebugView -and $Metadata.rt_debug_view -ne $Case.rtDebugView) {
        throw "capture metadata rt_debug_view was $($Metadata.rt_debug_view), expected $($Case.rtDebugView)."
    }
    if ($null -ne $Case.rtRestirDi -and [bool]$Metadata.rt_restir_di_enabled -ne [bool]$Case.rtRestirDi) {
        throw "capture metadata rt_restir_di_enabled was $($Metadata.rt_restir_di_enabled), expected $($Case.rtRestirDi)."
    }
    if ($null -ne $Case.rtRestirDiSpatial -and [bool]$Metadata.rt_restir_di_spatial_enabled -ne [bool]$Case.rtRestirDiSpatial) {
        throw "capture metadata rt_restir_di_spatial_enabled was $($Metadata.rt_restir_di_spatial_enabled), expected $($Case.rtRestirDiSpatial)."
    }
    if ($null -ne $Case.rtRestirDiSpatialSamples -and [int]$Metadata.rt_restir_di_spatial_sample_count -ne [int]$Case.rtRestirDiSpatialSamples) {
        throw "capture metadata rt_restir_di_spatial_sample_count was $($Metadata.rt_restir_di_spatial_sample_count), expected $($Case.rtRestirDiSpatialSamples)."
    }
    if ($null -ne $Case.rtRestirGi -and [bool]$Metadata.rt_restir_gi_enabled -ne [bool]$Case.rtRestirGi) {
        throw "capture metadata rt_restir_gi_enabled was $($Metadata.rt_restir_gi_enabled), expected $($Case.rtRestirGi)."
    }
    if ($null -ne $Case.rtRestirGiSpatial -and [bool]$Metadata.rt_restir_gi_spatial_enabled -ne [bool]$Case.rtRestirGiSpatial) {
        throw "capture metadata rt_restir_gi_spatial_enabled was $($Metadata.rt_restir_gi_spatial_enabled), expected $($Case.rtRestirGiSpatial)."
    }
    if ($null -ne $Case.rtRestirGiSpatialSamples -and [int]$Metadata.rt_restir_gi_spatial_sample_count -ne [int]$Case.rtRestirGiSpatialSamples) {
        throw "capture metadata rt_restir_gi_spatial_sample_count was $($Metadata.rt_restir_gi_spatial_sample_count), expected $($Case.rtRestirGiSpatialSamples)."
    }
    if ($null -ne $Case.rtTemporalDenoise -and [bool]$Metadata.rt_temporal_denoise_enabled -ne [bool]$Case.rtTemporalDenoise) {
        throw "capture metadata rt_temporal_denoise_enabled was $($Metadata.rt_temporal_denoise_enabled), expected $($Case.rtTemporalDenoise)."
    }
    if ($null -ne $Case.expectedRtFrameRendered) {
        Assert-MetadataBooleanField -Metadata $Metadata -FieldName "rt_frame_rendered" -ExpectedValue $Case.expectedRtFrameRendered
    }
    if ($null -ne $Case.expectedRtRestirDiRendered) {
        Assert-MetadataBooleanField -Metadata $Metadata -FieldName "rt_restir_di_rendered" -ExpectedValue $Case.expectedRtRestirDiRendered
    }
    if ($null -ne $Case.expectedRtRestirGiRendered) {
        Assert-MetadataBooleanField -Metadata $Metadata -FieldName "rt_restir_gi_rendered" -ExpectedValue $Case.expectedRtRestirGiRendered
    }
    if ($null -ne $Case.expectedRtRestirGiSpatialRendered) {
        Assert-MetadataBooleanField -Metadata $Metadata -FieldName "rt_restir_gi_spatial_rendered" -ExpectedValue $Case.expectedRtRestirGiSpatialRendered
    }
    if ($null -ne $Case.expectedRtResolveReady) {
        Assert-MetadataBooleanField -Metadata $Metadata -FieldName "rt_resolve_ready" -ExpectedValue $Case.expectedRtResolveReady
    }
    if ($Case.cameraPath -and $Metadata.cameraPath -ne $Case.cameraPath) {
        throw "capture metadata cameraPath was $($Metadata.cameraPath), expected $($Case.cameraPath)."
    }
    if ($Case.cameraPathCenter -and $Metadata.cameraPathCenter -ne $Case.cameraPathCenter) {
        throw "capture metadata cameraPathCenter was $($Metadata.cameraPathCenter), expected $($Case.cameraPathCenter)."
    }
    if ($null -ne $Case.cameraPathRadius -and [double]$Metadata.cameraPathRadius -ne [double]$Case.cameraPathRadius) {
        throw "capture metadata cameraPathRadius was $($Metadata.cameraPathRadius), expected $($Case.cameraPathRadius)."
    }
    if ($null -ne $Case.cameraPathHeight -and [double]$Metadata.cameraPathHeight -ne [double]$Case.cameraPathHeight) {
        throw "capture metadata cameraPathHeight was $($Metadata.cameraPathHeight), expected $($Case.cameraPathHeight)."
    }
    if ($null -ne $Case.cameraPathPeriodFrames -and [int64]$Metadata.cameraPathPeriodFrames -ne [int64]$Case.cameraPathPeriodFrames) {
        throw "capture metadata cameraPathPeriodFrames was $($Metadata.cameraPathPeriodFrames), expected $($Case.cameraPathPeriodFrames)."
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

function Test-CaseProperty {
    param(
        [Parameter(Mandatory = $true)]
        [object]$Case,
        [Parameter(Mandatory = $true)]
        [string]$FieldName
    )

    $Case.PSObject.Properties.Name -contains $FieldName
}

function Measure-PpmSignal {
    param(
        [Parameter(Mandatory = $true)]
        [string]$PpmPath
    )

    $bytes = [System.IO.File]::ReadAllBytes($PpmPath)
    $header = Read-PpmHeader -Bytes $bytes
    $pixelCount = $header.Width * $header.Height
    if ($pixelCount -le 0) {
        throw "PPM dimensions produced no pixels: $PpmPath"
    }
    $pixelEnd = $header.DataOffset + ($pixelCount * 3)
    if ($bytes.Length -lt $pixelEnd) {
        throw "PPM byte length was $($bytes.Length), expected at least $pixelEnd."
    }

    $nonZeroPixels = 0
    $redPixels = 0
    $greenPixels = 0
    $bluePixels = 0
    $cyanPixels = 0
    $magentaPixels = 0
    $yellowPixels = 0
    $whitePixels = 0
    $minRgb = 255
    $maxRgb = 0
    for ($i = $header.DataOffset; $i -lt $pixelEnd; $i += 3) {
        $r = [int]$bytes[$i]
        $g = [int]$bytes[$i + 1]
        $b = [int]$bytes[$i + 2]
        if ($r -ne 0 -or $g -ne 0 -or $b -ne 0) {
            $nonZeroPixels++
        }
        if ($r -gt 0 -and $g -eq 0 -and $b -eq 0) {
            $redPixels++
        }
        if ($r -eq 0 -and $g -gt 0 -and $b -eq 0) {
            $greenPixels++
        }
        if ($r -eq 0 -and $g -eq 0 -and $b -gt 0) {
            $bluePixels++
        }
        if ($r -eq 0 -and $g -gt 0 -and $b -gt 0) {
            $cyanPixels++
        }
        if ($r -gt 0 -and $g -eq 0 -and $b -gt 0) {
            $magentaPixels++
        }
        if ($r -gt 0 -and $g -gt 0 -and $b -eq 0) {
            $yellowPixels++
        }
        if ($r -gt 0 -and $r -eq $g -and $g -eq $b) {
            $whitePixels++
        }
        $minRgb = [Math]::Min($minRgb, [Math]::Min($r, [Math]::Min($g, $b)))
        $maxRgb = [Math]::Max($maxRgb, [Math]::Max($r, [Math]::Max($g, $b)))
    }

    [PSCustomObject]@{
        PixelCount = $pixelCount
        NonZeroPixels = $nonZeroPixels
        NonZeroPixelRatio = [double]$nonZeroPixels / [double]$pixelCount
        MinRgb = $minRgb
        MaxRgb = $maxRgb
        RgbRange = $maxRgb - $minRgb
        ColorPixelRatios = [PSCustomObject]@{
            red = [double]$redPixels / [double]$pixelCount
            green = [double]$greenPixels / [double]$pixelCount
            blue = [double]$bluePixels / [double]$pixelCount
            cyan = [double]$cyanPixels / [double]$pixelCount
            magenta = [double]$magentaPixels / [double]$pixelCount
            yellow = [double]$yellowPixels / [double]$pixelCount
            white = [double]$whitePixels / [double]$pixelCount
        }
    }
}

function Assert-PpmColorRatios {
    param(
        [Parameter(Mandatory = $true)]
        [object]$Signal,
        [Parameter(Mandatory = $true)]
        [object]$Case
    )

    if (-not (Test-CaseProperty -Case $Case -FieldName "expectedMaxColorPixelRatios")) {
        return
    }
    if ($null -eq $Case.expectedMaxColorPixelRatios) {
        throw "PPM signal threshold expectedMaxColorPixelRatios was null for $($Case.name)."
    }

    foreach ($expected in $Case.expectedMaxColorPixelRatios.PSObject.Properties) {
        $colorName = $expected.Name
        if (-not ($Signal.ColorPixelRatios.PSObject.Properties.Name -contains $colorName)) {
            throw "PPM signal threshold expectedMaxColorPixelRatios used unsupported color $colorName for $($Case.name)."
        }

        $actualRatio = [double]$Signal.ColorPixelRatios.PSObject.Properties[$colorName].Value
        $maxRatio = [double]$expected.Value
        if ($actualRatio -gt $maxRatio) {
            throw "PPM $colorName pixel ratio $actualRatio exceeded expected maximum $maxRatio for $($Case.name)."
        }
    }
}

function Assert-PpmSignal {
    param(
        [Parameter(Mandatory = $true)]
        [string]$PpmPath,
        [Parameter(Mandatory = $true)]
        [object]$Case
    )

    $signal = Measure-PpmSignal -PpmPath $PpmPath
    if ($signal.NonZeroPixels -le 0) {
        throw "PPM capture contains only zero RGB pixels: $PpmPath"
    }
    if (Test-CaseProperty -Case $Case -FieldName "expectedMinNonZeroPixelRatio") {
        if ($null -eq $Case.expectedMinNonZeroPixelRatio) {
            throw "PPM signal threshold expectedMinNonZeroPixelRatio was null for $($Case.name)."
        }
        $minRatio = [double]$Case.expectedMinNonZeroPixelRatio
        if ($signal.NonZeroPixelRatio -lt $minRatio) {
            throw "PPM non-zero pixel ratio $($signal.NonZeroPixelRatio) was below expected minimum $minRatio for $($Case.name)."
        }
    }
    if (Test-CaseProperty -Case $Case -FieldName "expectedMinRgbRange") {
        if ($null -eq $Case.expectedMinRgbRange) {
            throw "PPM signal threshold expectedMinRgbRange was null for $($Case.name)."
        }
        $minRange = [int]$Case.expectedMinRgbRange
        if ($signal.RgbRange -lt $minRange) {
            throw "PPM RGB range $($signal.RgbRange) was below expected minimum $minRange for $($Case.name)."
        }
    }
    Assert-PpmColorRatios -Signal $signal -Case $Case
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
    "REVOLUMETRIC_CAPTURE_FRAMES",
    "REVOLUMETRIC_CAPTURE_DIR",
    "REVOLUMETRIC_CAPTURE_PREFIX",
    "REVOLUMETRIC_RENDER_MODE",
    "REVOLUMETRIC_DENOISER",
    "REVOLUMETRIC_VPT_DEBUG_VIEW",
    "REVOLUMETRIC_RT_DEBUG_VIEW",
    "REVOLUMETRIC_RT_RESTIR_DI",
    "REVOLUMETRIC_RT_RESTIR_DI_SPATIAL",
    "REVOLUMETRIC_RT_RESTIR_DI_SPATIAL_SAMPLES",
    "REVOLUMETRIC_RT_RESTIR_GI",
    "REVOLUMETRIC_RT_RESTIR_GI_SPATIAL",
    "REVOLUMETRIC_RT_RESTIR_GI_SPATIAL_SAMPLES",
    "REVOLUMETRIC_RT_TEMPORAL_DENOISE",
    "REVOLUMETRIC_CAMERA_PATH",
    "REVOLUMETRIC_CAMERA_PATH_CENTER",
    "REVOLUMETRIC_CAMERA_PATH_RADIUS",
    "REVOLUMETRIC_CAMERA_PATH_HEIGHT",
    "REVOLUMETRIC_CAMERA_PATH_PERIOD_FRAMES",
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
        if ($case.requiresRt -and -not $Rt) {
            Write-Host "skip visual baseline $($case.name): requires -Rt"
            continue
        }

        $caseOutputDir = Join-Path $OutputDir $case.name
        Remove-Item -LiteralPath $caseOutputDir -Recurse -Force -ErrorAction SilentlyContinue
        New-Item -ItemType Directory -Force $caseOutputDir | Out-Null

        $env:REVOLUMETRIC_CAPTURE_FRAME = "$captureFrame"
        Remove-Item Env:\REVOLUMETRIC_CAPTURE_FRAMES -ErrorAction SilentlyContinue
        $env:REVOLUMETRIC_CAPTURE_DIR = $caseOutputDir
        $env:REVOLUMETRIC_CAPTURE_PREFIX = $case.name
        $env:REVOLUMETRIC_EXIT_AFTER_FRAMES = "$frames"
        if ($case.renderMode) {
            $env:REVOLUMETRIC_RENDER_MODE = $case.renderMode
        } else {
            $env:REVOLUMETRIC_RENDER_MODE = "vpt"
        }
        $env:REVOLUMETRIC_DENOISER = $case.denoiser
        if ($case.debugView -eq "final") {
            Remove-Item Env:\REVOLUMETRIC_VPT_DEBUG_VIEW -ErrorAction SilentlyContinue
        } else {
            $env:REVOLUMETRIC_VPT_DEBUG_VIEW = $case.debugView
        }
        if ($case.rtDebugView) {
            $env:REVOLUMETRIC_RT_DEBUG_VIEW = $case.rtDebugView
        } else {
            Remove-Item Env:\REVOLUMETRIC_RT_DEBUG_VIEW -ErrorAction SilentlyContinue
        }
        if ($null -ne $case.rtRestirDi) {
            $env:REVOLUMETRIC_RT_RESTIR_DI = ([string]$case.rtRestirDi).ToLowerInvariant()
        } else {
            Remove-Item Env:\REVOLUMETRIC_RT_RESTIR_DI -ErrorAction SilentlyContinue
        }
        if ($null -ne $case.rtRestirDiSpatial) {
            $env:REVOLUMETRIC_RT_RESTIR_DI_SPATIAL = ([string]$case.rtRestirDiSpatial).ToLowerInvariant()
        } else {
            Remove-Item Env:\REVOLUMETRIC_RT_RESTIR_DI_SPATIAL -ErrorAction SilentlyContinue
        }
        if ($null -ne $case.rtRestirDiSpatialSamples) {
            $env:REVOLUMETRIC_RT_RESTIR_DI_SPATIAL_SAMPLES = "$($case.rtRestirDiSpatialSamples)"
        } else {
            Remove-Item Env:\REVOLUMETRIC_RT_RESTIR_DI_SPATIAL_SAMPLES -ErrorAction SilentlyContinue
        }
        if ($null -ne $case.rtRestirGi) {
            $env:REVOLUMETRIC_RT_RESTIR_GI = ([string]$case.rtRestirGi).ToLowerInvariant()
        } else {
            Remove-Item Env:\REVOLUMETRIC_RT_RESTIR_GI -ErrorAction SilentlyContinue
        }
        if ($null -ne $case.rtRestirGiSpatial) {
            $env:REVOLUMETRIC_RT_RESTIR_GI_SPATIAL = ([string]$case.rtRestirGiSpatial).ToLowerInvariant()
        } else {
            Remove-Item Env:\REVOLUMETRIC_RT_RESTIR_GI_SPATIAL -ErrorAction SilentlyContinue
        }
        if ($null -ne $case.rtRestirGiSpatialSamples) {
            $env:REVOLUMETRIC_RT_RESTIR_GI_SPATIAL_SAMPLES = "$($case.rtRestirGiSpatialSamples)"
        } else {
            Remove-Item Env:\REVOLUMETRIC_RT_RESTIR_GI_SPATIAL_SAMPLES -ErrorAction SilentlyContinue
        }
        if ($null -ne $case.rtTemporalDenoise) {
            $env:REVOLUMETRIC_RT_TEMPORAL_DENOISE = ([string]$case.rtTemporalDenoise).ToLowerInvariant()
        } else {
            Remove-Item Env:\REVOLUMETRIC_RT_TEMPORAL_DENOISE -ErrorAction SilentlyContinue
        }
        if ($case.cameraPath) {
            $env:REVOLUMETRIC_CAMERA_PATH = $case.cameraPath
        } else {
            Remove-Item Env:\REVOLUMETRIC_CAMERA_PATH -ErrorAction SilentlyContinue
        }
        if ($case.cameraPathCenter) {
            $env:REVOLUMETRIC_CAMERA_PATH_CENTER = $case.cameraPathCenter
        } else {
            Remove-Item Env:\REVOLUMETRIC_CAMERA_PATH_CENTER -ErrorAction SilentlyContinue
        }
        if ($null -ne $case.cameraPathRadius) {
            $env:REVOLUMETRIC_CAMERA_PATH_RADIUS = "$($case.cameraPathRadius)"
        } else {
            Remove-Item Env:\REVOLUMETRIC_CAMERA_PATH_RADIUS -ErrorAction SilentlyContinue
        }
        if ($null -ne $case.cameraPathHeight) {
            $env:REVOLUMETRIC_CAMERA_PATH_HEIGHT = "$($case.cameraPathHeight)"
        } else {
            Remove-Item Env:\REVOLUMETRIC_CAMERA_PATH_HEIGHT -ErrorAction SilentlyContinue
        }
        if ($null -ne $case.cameraPathPeriodFrames) {
            $env:REVOLUMETRIC_CAMERA_PATH_PERIOD_FRAMES = "$($case.cameraPathPeriodFrames)"
        } else {
            Remove-Item Env:\REVOLUMETRIC_CAMERA_PATH_PERIOD_FRAMES -ErrorAction SilentlyContinue
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
        Assert-PpmSignal -PpmPath $ppmPath -Case $case
    }
} finally {
    Set-Location $previousLocation
    foreach ($entry in $previousEnv.GetEnumerator()) {
        [Environment]::SetEnvironmentVariable($entry.Key, $entry.Value, "Process")
    }
}
