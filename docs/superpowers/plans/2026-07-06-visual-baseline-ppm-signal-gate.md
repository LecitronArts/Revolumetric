# Visual Baseline PPM Signal Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the visual baseline validator reject blank, almost blank, and flat PPM captures by checking pixel coverage and RGB range.

**Architecture:** Keep the gate in the existing PowerShell visual baseline script and manifest. Source-contract tests lock the script helper names and manifest fields so future edits cannot silently fall back to a one-byte non-zero check.

**Tech Stack:** Rust 2024 source-contract tests, PowerShell 5+ script validation, JSON manifest thresholds, Cargo fmt/test/clippy.

---

## File Structure

- Modify `src/render/source_checks.rs`: update the visual baseline source-contract tests so they require the new PPM signal helper, signal assertion, and manifest threshold fields.
- Modify `run/validate-visual-baseline.ps1`: replace `Assert-PpmHasNonZeroRgb` with `Measure-PpmSignal` and `Assert-PpmSignal`; compute per-pixel signal ratio and RGB range; enforce optional manifest thresholds.
- Modify `run/visual-baselines.json`: add conservative measured thresholds to `svgf_final` and `rt_surface_debug`.
- No renderer, shader, capture metadata, or UI code changes are part of this slice.

## Task 1: Source-Contract Tests For PPM Signal Gate

**Files:**
- Modify: `src/render/source_checks.rs`

- [ ] **Step 1: Write the failing script source-check expectations**

In `visual_baseline_script_validates_captures_metadata_and_nonblank_ppm`, replace the `"Assert-PpmHasNonZeroRgb"` token with these tokens:

```rust
"Measure-PpmSignal",
"Assert-PpmSignal",
"NonZeroPixelRatio",
"RgbRange",
"expectedMinNonZeroPixelRatio",
"expectedMinRgbRange",
```

The relevant assertion array should contain:

```rust
&[
    "[string]$Manifest",
    "visual-baselines.json",
    "REVOLUMETRIC_CAPTURE_FRAME",
    "REVOLUMETRIC_CAPTURE_DIR",
    "REVOLUMETRIC_CAPTURE_PREFIX",
    "REVOLUMETRIC_RENDER_MODE",
    "ConvertFrom-Json",
    "Assert-CaptureMetadata",
    "Assert-MetadataBooleanField",
    "$Metadata.PSObject.Properties.Name -contains $FieldName",
    "Assert-PpmMatchesMetadata",
    "Measure-PpmSignal",
    "Assert-PpmSignal",
    "NonZeroPixelRatio",
    "RgbRange",
    "expectedMinNonZeroPixelRatio",
    "expectedMinRgbRange",
    "[switch]$Rt",
    "requiresRt",
    "REVOLUMETRIC_RT_DEBUG_VIEW",
    "REVOLUMETRIC_RT_RESTIR_DI",
    "REVOLUMETRIC_RT_RESTIR_DI_SPATIAL",
    "REVOLUMETRIC_RT_RESTIR_DI_SPATIAL_SAMPLES",
    "REVOLUMETRIC_RT_RESTIR_GI",
    "REVOLUMETRIC_RT_TEMPORAL_DENOISE",
    "render_backend",
    "render_mode",
    "rt_debug_view",
    "rt_restir_di_enabled",
    "rt_restir_di_spatial_enabled",
    "rt_restir_di_spatial_sample_count",
    "rt_restir_gi_enabled",
    "rt_temporal_denoise_enabled",
    "rt_frame_rendered",
    "rt_restir_di_rendered",
    "rt_restir_gi_rendered",
    "rt_resolve_ready",
    ".\\run\\validate-nrd.ps1",
    "cargo run --features desktop --bin revolumetric",
]
```

- [ ] **Step 2: Write the failing manifest source-check expectations**

In `visual_baseline_manifest_covers_svgf_and_reblur_debug_cases`, add these tokens to the manifest expectation array:

```rust
"\"expectedMinNonZeroPixelRatio\": 0.25",
"\"expectedMinRgbRange\": 32",
```

Keep the existing RT active metadata tokens unchanged.

- [ ] **Step 3: Run RED**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib visual_baseline
```

Expected result: both source-check tests fail because the script and manifest do not yet contain `Measure-PpmSignal`, `Assert-PpmSignal`, `expectedMinNonZeroPixelRatio`, or `expectedMinRgbRange`.

## Task 2: Manifest Thresholds For Measured Captures

**Files:**
- Modify: `run/visual-baselines.json`

- [ ] **Step 1: Add thresholds to `svgf_final`**

Add these fields to the `svgf_final` case after `expectedEffectiveDenoiser`:

```json
"expectedMinNonZeroPixelRatio": 0.25,
"expectedMinRgbRange": 32
```

- [ ] **Step 2: Add thresholds to `rt_surface_debug`**

Add these fields to the `rt_surface_debug` case after `expectedEffectiveDenoiser`:

```json
"expectedMinNonZeroPixelRatio": 0.25,
"expectedMinRgbRange": 32,
```

Do not add these fields to `reblur_final` or `reblur_nrd_validation` in this task because those NRD-gated captures were not measured in this environment.

- [ ] **Step 3: Run partial GREEN**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib visual_baseline_manifest_covers_svgf_and_reblur_debug_cases
```

Expected result: the manifest coverage test passes, while the script coverage test still fails until Task 3 implements the validator.

## Task 3: Validator PPM Signal Measurement

**Files:**
- Modify: `run/validate-visual-baseline.ps1`

- [ ] **Step 1: Replace the byte-level non-zero helper**

Delete `Assert-PpmHasNonZeroRgb` and add this helper in the same location, after `Assert-PpmMatchesMetadata`:

```powershell
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

    $nonZeroPixels = 0
    $minRgb = 255
    $maxRgb = 0
    for ($i = $header.DataOffset; $i -lt $bytes.Length; $i += 3) {
        $r = [int]$bytes[$i]
        $g = [int]$bytes[$i + 1]
        $b = [int]$bytes[$i + 2]
        if ($r -ne 0 -or $g -ne 0 -or $b -ne 0) {
            $nonZeroPixels++
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
    }
}
```

- [ ] **Step 2: Add threshold assertion helper**

Add this function immediately after `Measure-PpmSignal`:

```powershell
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
    if ($null -ne $Case.expectedMinNonZeroPixelRatio) {
        $minRatio = [double]$Case.expectedMinNonZeroPixelRatio
        if ($signal.NonZeroPixelRatio -lt $minRatio) {
            throw "PPM non-zero pixel ratio $($signal.NonZeroPixelRatio) was below expected minimum $minRatio for $($Case.name)."
        }
    }
    if ($null -ne $Case.expectedMinRgbRange) {
        $minRange = [int]$Case.expectedMinRgbRange
        if ($signal.RgbRange -lt $minRange) {
            throw "PPM RGB range $($signal.RgbRange) was below expected minimum $minRange for $($Case.name)."
        }
    }
}
```

- [ ] **Step 3: Wire the assertion into each case**

Replace:

```powershell
Assert-PpmHasNonZeroRgb -PpmPath $ppmPath
```

with:

```powershell
Assert-PpmSignal -PpmPath $ppmPath -Case $case
```

- [ ] **Step 4: Run GREEN for focused source checks**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib visual_baseline
```

Expected result: both focused source-check tests pass.

## Task 4: Verification, Review, And Commit

**Files:**
- Review only: `src/render/source_checks.rs`, `run/validate-visual-baseline.ps1`, `run/visual-baselines.json`

- [ ] **Step 1: Format and source-check**

Run:

```powershell
cargo fmt --check
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib visual_baseline
```

Expected result: formatting check passes and focused source-check tests pass.

- [ ] **Step 2: Run full Rust verification**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo clippy --all-targets -- -D warnings
$env:REVOLUMETRIC_SHADER_COMPILE='strict'; cargo test --lib
```

Expected result: each command exits with code 0.

- [ ] **Step 3: Run real visual baseline validation**

Run:

```powershell
.\run\validate-visual-baseline.ps1 -Rt
```

Expected result: `svgf_final` and `rt_surface_debug` pass the new signal thresholds. NRD cases may still be skipped unless `-Nrd` is provided.

- [ ] **Step 4: Check diff hygiene**

Run:

```powershell
git diff --check
git status --short --branch
```

Expected result: `git diff --check` exits with code 0. Windows CRLF warnings are acceptable if exit code is 0.

- [ ] **Step 5: Request read-only code review**

Ask a reviewer to inspect only this slice:

```text
Review the visual baseline PPM signal gate changes. Check that the validator measures per-pixel signal correctly, optional manifest thresholds are enforced only when present, the default one-byte non-zero gate is gone, and the thresholds are conservative for the measured captures. Do not edit files.
```

Fix any Critical or Important issues, then rerun the relevant verification command.

- [ ] **Step 6: Commit implementation**

Run:

```powershell
git add src/render/source_checks.rs run/validate-visual-baseline.ps1 run/visual-baselines.json
git commit -m "feat: strengthen visual baseline PPM signal gate"
```
