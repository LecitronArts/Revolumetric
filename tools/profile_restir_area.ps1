param(
    [int]$Frames = 120,
    [int]$WarmupFrames = 20,
    [string]$Csv = "target\profile-restir-area.csv",
    [switch]$SkipStrictBuild,
    [switch]$NoDirectSpatial
)

$ErrorActionPreference = "Stop"

if ($Frames -le 0) {
    throw "Frames must be positive."
}
if ($WarmupFrames -lt 0 -or $WarmupFrames -ge $Frames) {
    throw "WarmupFrames must be >= 0 and < Frames."
}

$previousEnv = @{}
foreach ($name in @(
    "REVOLUMETRIC_SHADER_COMPILE",
    "CARGO_TARGET_DIR",
    "REVOLUMETRIC_GPU_PROFILER",
    "REVOLUMETRIC_GPU_PROFILE_CSV",
    "REVOLUMETRIC_GPU_PROFILE_CSV_FLUSH_INTERVAL",
    "REVOLUMETRIC_VPT_RESTIR_DI",
    "REVOLUMETRIC_RESTIR_DI_SPATIAL",
    "REVOLUMETRIC_AREA_RESTIR",
    "REVOLUMETRIC_EXIT_AFTER_FRAMES"
)) {
    $previousEnv[$name] = [Environment]::GetEnvironmentVariable($name, "Process")
}

try {
    $env:REVOLUMETRIC_SHADER_COMPILE = "strict"
    $env:CARGO_TARGET_DIR = "target\codex-strict"

    if (-not $SkipStrictBuild) {
        cargo build --bin revolumetric
        if ($LASTEXITCODE -ne 0) {
            throw "strict cargo build failed."
        }
    }

    Remove-Item $Csv -ErrorAction SilentlyContinue

    $env:REVOLUMETRIC_GPU_PROFILER = "on"
    $env:REVOLUMETRIC_GPU_PROFILE_CSV = $Csv
    $env:REVOLUMETRIC_GPU_PROFILE_CSV_FLUSH_INTERVAL = "1"
    $env:REVOLUMETRIC_VPT_RESTIR_DI = "on"
    $env:REVOLUMETRIC_AREA_RESTIR = "on"
    $env:REVOLUMETRIC_EXIT_AFTER_FRAMES = "$Frames"
    if ($NoDirectSpatial) {
        $env:REVOLUMETRIC_RESTIR_DI_SPATIAL = "off"
    } else {
        $env:REVOLUMETRIC_RESTIR_DI_SPATIAL = "on"
    }

    cargo run --bin revolumetric
    if ($LASTEXITCODE -ne 0) {
        throw "profile run failed."
    }

    if (-not (Test-Path $Csv)) {
        throw "profile CSV was not written: $Csv"
    }

    $rows = Import-Csv $Csv | Where-Object { [int]$_.frame -gt $WarmupFrames }
    if ($rows.Count -eq 0) {
        throw "no profile rows remain after warmup filter."
    }

    Write-Host "profile=$Csv frames=$Frames warmup=$WarmupFrames direct_spatial=$(-not $NoDirectSpatial)"
    foreach ($column in @(
        "vpt_surface_bootstrap_ms",
        "vpt_surface_selected_ms",
        "vpt_ms",
        "restir_di_initial_ms",
        "restir_di_temporal_ms",
        "restir_di_spatial_ms",
        "area_restir_initial_ms",
        "area_restir_temporal_ms",
        "area_restir_spatial_ms",
        "vpt_temporal_ms",
        "vpt_atrous_ms",
        "vpt_nrd_confidence_ms",
        "vpt_nrd_frontend_ms",
        "vpt_nrd_adapter_ms",
        "vpt_nrd_resolve_ms",
        "postprocess_ms",
        "blit_to_swapchain_ms",
        "total_ms"
    )) {
        $values = $rows | ForEach-Object { [double]$_.$column } | Sort-Object
        $avg = ($values | Measure-Object -Average).Average
        $median = $values[[int]($values.Count / 2)]
        $p95Index = [int][Math]::Min($values.Count - 1, [Math]::Floor($values.Count * 0.95))
        $p95 = $values[$p95Index]
        "{0}: avg={1:N4} median={2:N4} p95={3:N4}" -f $column, $avg, $median, $p95
    }
} finally {
    foreach ($entry in $previousEnv.GetEnumerator()) {
        [Environment]::SetEnvironmentVariable($entry.Key, $entry.Value, "Process")
    }
}
