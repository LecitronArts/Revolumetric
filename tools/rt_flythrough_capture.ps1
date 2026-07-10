param(
    [ValidateSet("auto", "rt", "vpt")]
    [string]$Mode = "rt",
    [int]$Frames = 0,
    [int]$CaptureFrame = -1,
    [string]$CaptureFrames = "",
    [string]$CaptureDir = "target\captures",
    [string]$CapturePrefix = "rt_flythrough",
    [string]$Center = "64,32,64",
    [double]$Radius = 40,
    [double]$Height = 36,
    [int]$PeriodFrames = 240,
    [string]$RtDebugView = ""
)

$ErrorActionPreference = "Stop"

if ($Frames -lt 0) {
    throw "Frames must be >= 0. Use 0 for real-time observation without automatic exit."
}
if ($CaptureFrame -lt -1) {
    throw "CaptureFrame must be >= 0, or -1 to disable capture."
}
$parsedCaptureFrames = @()
if ($CaptureFrames.Trim().Length -gt 0) {
    foreach ($part in $CaptureFrames.Split(",")) {
        $trimmed = $part.Trim()
        $frameValue = 0
        if ($trimmed.Length -eq 0 -or -not [int]::TryParse($trimmed, [ref]$frameValue) -or $frameValue -lt 0) {
            throw "CaptureFrames must be a comma-separated list of non-negative integers."
        }
        $parsedCaptureFrames += $frameValue
    }
}
if ($Radius -le 0) {
    throw "Radius must be positive."
}
if ($PeriodFrames -le 0) {
    throw "PeriodFrames must be positive."
}

$previousEnv = @{}
foreach ($name in @(
    "REVOLUMETRIC_CAMERA_PATH",
    "REVOLUMETRIC_CAMERA_PATH_CENTER",
    "REVOLUMETRIC_CAMERA_PATH_RADIUS",
    "REVOLUMETRIC_CAMERA_PATH_HEIGHT",
    "REVOLUMETRIC_CAMERA_PATH_PERIOD_FRAMES",
    "REVOLUMETRIC_RENDER_MODE",
    "REVOLUMETRIC_EXIT_AFTER_FRAMES",
    "REVOLUMETRIC_CAPTURE_FRAME",
    "REVOLUMETRIC_CAPTURE_FRAMES",
    "REVOLUMETRIC_CAPTURE_DIR",
    "REVOLUMETRIC_CAPTURE_PREFIX",
    "REVOLUMETRIC_RT_DEBUG_VIEW"
)) {
    $previousEnv[$name] = [Environment]::GetEnvironmentVariable($name, "Process")
}

try {
    $env:REVOLUMETRIC_CAMERA_PATH = "gallery"
    $env:REVOLUMETRIC_CAMERA_PATH_CENTER = $Center
    $env:REVOLUMETRIC_CAMERA_PATH_RADIUS = "$Radius"
    $env:REVOLUMETRIC_CAMERA_PATH_HEIGHT = "$Height"
    $env:REVOLUMETRIC_CAMERA_PATH_PERIOD_FRAMES = "$PeriodFrames"
    $env:REVOLUMETRIC_RENDER_MODE = $Mode

    if ($RtDebugView.Trim().Length -gt 0) {
        $env:REVOLUMETRIC_RT_DEBUG_VIEW = $RtDebugView
    } else {
        Remove-Item Env:\REVOLUMETRIC_RT_DEBUG_VIEW -ErrorAction SilentlyContinue
    }

    $effectiveFrames = $Frames
    $maxCaptureFrame = $CaptureFrame
    foreach ($frame in $parsedCaptureFrames) {
        if ($frame -gt $maxCaptureFrame) {
            $maxCaptureFrame = $frame
        }
    }

    if ($CaptureFrame -ge 0) {
        $env:REVOLUMETRIC_CAPTURE_FRAME = "$CaptureFrame"
    } else {
        Remove-Item Env:\REVOLUMETRIC_CAPTURE_FRAME -ErrorAction SilentlyContinue
    }
    if ($parsedCaptureFrames.Count -gt 0) {
        $env:REVOLUMETRIC_CAPTURE_FRAMES = ($parsedCaptureFrames -join ",")
    } else {
        Remove-Item Env:\REVOLUMETRIC_CAPTURE_FRAMES -ErrorAction SilentlyContinue
    }

    if ($maxCaptureFrame -ge 0) {
        $env:REVOLUMETRIC_CAPTURE_DIR = $CaptureDir
        $env:REVOLUMETRIC_CAPTURE_PREFIX = $CapturePrefix
        if ($effectiveFrames -le $maxCaptureFrame) {
            $effectiveFrames = $maxCaptureFrame + 1
        }
    } else {
        Remove-Item Env:\REVOLUMETRIC_CAPTURE_DIR -ErrorAction SilentlyContinue
        Remove-Item Env:\REVOLUMETRIC_CAPTURE_PREFIX -ErrorAction SilentlyContinue
    }

    if ($effectiveFrames -gt 0) {
        $env:REVOLUMETRIC_EXIT_AFTER_FRAMES = "$effectiveFrames"
    } else {
        Remove-Item Env:\REVOLUMETRIC_EXIT_AFTER_FRAMES -ErrorAction SilentlyContinue
    }

    cargo run --features desktop --bin revolumetric
    if ($LASTEXITCODE -ne 0) {
        throw "flythrough run failed."
    }
} finally {
    foreach ($entry in $previousEnv.GetEnumerator()) {
        [Environment]::SetEnvironmentVariable($entry.Key, $entry.Value, "Process")
    }
}
