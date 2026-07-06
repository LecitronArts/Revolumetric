# RT Visual Baseline Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Record requested render mode, actual capture backend, and RT controls in visual baseline metadata, then validate those fields from the local baseline manifest/script.

**Architecture:** Keep the current VPT image capture path intact and extend its metadata contract. Thread `RuntimeSettings.rt` into `VptFrameInputs` so VPT fallback captures can still record requested RT controls. Update the baseline manifest and PowerShell validator to set and assert RT case fields without claiming hardware RT image readback is implemented.

**Tech Stack:** Rust renderer modules, Rust unit/source-contract tests, PowerShell validation script, JSON visual baseline manifest, Cargo test/clippy/fmt.

---

## File Structure

- Modify `src/render/capture.rs`: add capture metadata fields and serialization test assertions.
- Modify `src/render/vpt_pipeline.rs`: add `RtSettings` to `VptFrameInputs`, add render-mode/RT-debug name helpers, populate metadata fields, and add helper tests.
- Modify `src/render/runtime.rs`: pass `input.settings.rt` into VPT frame inputs and update source-contract tests.
- Modify `src/render/source_checks.rs`: add source-contract checks for RT baseline manifest and validation script support.
- Modify `run/visual-baselines.json`: add one RT fallback contract case.
- Modify `run/validate-visual-baseline.ps1`: preserve/set/restore RT env vars and assert optional metadata fields.

---

### Task 1: Capture Metadata Contract

**Files:**
- Modify: `src/render/capture.rs`

- [ ] **Step 1: Write the failing serialization assertions**

In `metadata_json_records_frame_settings_and_paths`, add these fields to the `CaptureMetadata` literal:

```rust
render_backend: "vpt",
render_mode: "rt",
rt_debug_view: "surface",
rt_restir_di_enabled: true,
rt_restir_di_spatial_enabled: true,
rt_restir_di_spatial_sample_count: 4,
rt_restir_gi_enabled: true,
rt_temporal_denoise_enabled: true,
```

Add these assertions after the existing path/source assertions:

```rust
assert!(json.contains("\"render_backend\": \"vpt\""));
assert!(json.contains("\"render_mode\": \"rt\""));
assert!(json.contains("\"rt_debug_view\": \"surface\""));
assert!(json.contains("\"rt_restir_di_enabled\": true"));
assert!(json.contains("\"rt_restir_di_spatial_enabled\": true"));
assert!(json.contains("\"rt_restir_di_spatial_sample_count\": 4"));
assert!(json.contains("\"rt_restir_gi_enabled\": true"));
assert!(json.contains("\"rt_temporal_denoise_enabled\": true"));
```

- [ ] **Step 2: Run test to verify RED**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'
cargo test --lib render::capture::tests::metadata_json_records_frame_settings_and_paths
```

Expected: FAIL because `CaptureMetadata` has no backend/RT fields.

- [ ] **Step 3: Implement metadata fields and JSON output**

Add fields to `CaptureMetadata`:

```rust
pub render_backend: &'static str,
pub render_mode: &'static str,
pub rt_debug_view: &'static str,
pub rt_restir_di_enabled: bool,
pub rt_restir_di_spatial_enabled: bool,
pub rt_restir_di_spatial_sample_count: u32,
pub rt_restir_gi_enabled: bool,
pub rt_temporal_denoise_enabled: bool,
```

Add matching JSON lines immediately after `json_path`:

```rust
"  \"render_backend\": \"{}\",\n",
"  \"render_mode\": \"{}\",\n",
"  \"rt_debug_view\": \"{}\",\n",
"  \"rt_restir_di_enabled\": {},\n",
"  \"rt_restir_di_spatial_enabled\": {},\n",
"  \"rt_restir_di_spatial_sample_count\": {},\n",
"  \"rt_restir_gi_enabled\": {},\n",
"  \"rt_temporal_denoise_enabled\": {},\n",
```

Pass these values in the same order:

```rust
json_escape(self.render_backend),
json_escape(self.render_mode),
json_escape(self.rt_debug_view),
self.rt_restir_di_enabled,
self.rt_restir_di_spatial_enabled,
self.rt_restir_di_spatial_sample_count,
self.rt_restir_gi_enabled,
self.rt_temporal_denoise_enabled,
```

- [ ] **Step 4: Run test to verify GREEN**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'
cargo test --lib render::capture::tests::metadata_json_records_frame_settings_and_paths
```

Expected: PASS.

---

### Task 2: VPT Fallback Metadata Threading

**Files:**
- Modify: `src/render/vpt_pipeline.rs`
- Modify: `src/render/runtime.rs`

- [ ] **Step 1: Write failing helper and source-contract tests**

In `src/render/vpt_pipeline.rs`, add tests in the existing test module:

```rust
#[test]
fn capture_render_mode_name_uses_manifest_values() {
    assert_eq!(capture_render_mode_name(RenderMode::Auto), "auto");
    assert_eq!(capture_render_mode_name(RenderMode::Vpt), "vpt");
    assert_eq!(capture_render_mode_name(RenderMode::Rt), "rt");
}

#[test]
fn capture_rt_debug_view_name_uses_env_values() {
    assert_eq!(capture_rt_debug_view_name(RtDebugView::Off), "off");
    assert_eq!(capture_rt_debug_view_name(RtDebugView::Surface), "surface");
    assert_eq!(
        capture_rt_debug_view_name(RtDebugView::DirectReservoir),
        "direct_reservoir"
    );
}
```

In `src/render/runtime.rs`, extend `render_runtime_wires_capture_to_vpt_pipeline_only` to require:

```rust
"rt_settings: input.settings.rt",
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'
cargo test --lib render::vpt_pipeline::tests::capture_render_mode_name_uses_manifest_values render::vpt_pipeline::tests::capture_rt_debug_view_name_uses_env_values render::runtime::tests::render_runtime_wires_capture_to_vpt_pipeline_only
```

Expected: FAIL because helper functions and `VptFrameInputs::rt_settings` do not exist.

- [ ] **Step 3: Implement helper functions and threading**

In `src/render/vpt_pipeline.rs`, import `RtDebugView`:

```rust
use crate::render::rt_settings::{RtDebugView, RtSettings};
```

Add field to `VptFrameInputs`:

```rust
pub rt_settings: RtSettings,
```

Add helpers near `vpt_debug_view_name`:

```rust
fn capture_render_mode_name(render_mode: RenderMode) -> &'static str {
    match render_mode {
        RenderMode::Auto => "auto",
        RenderMode::Vpt => "vpt",
        RenderMode::Rt => "rt",
    }
}

fn capture_rt_debug_view_name(debug_view: RtDebugView) -> &'static str {
    match debug_view {
        RtDebugView::Off => "off",
        RtDebugView::Surface => "surface",
        RtDebugView::HitDistance => "hit_distance",
        RtDebugView::HistoryValid => "history_valid",
        RtDebugView::DirectReservoir => "direct_reservoir",
        RtDebugView::IndirectReservoir => "indirect_reservoir",
        RtDebugView::Temporal => "temporal",
    }
}
```

Populate `CaptureMetadata` in the VPT capture block:

```rust
render_backend: "vpt",
render_mode: capture_render_mode_name(inputs.lighting_settings.render_mode),
rt_debug_view: capture_rt_debug_view_name(inputs.rt_settings.debug_view),
rt_restir_di_enabled: inputs.rt_settings.restir_di_enabled,
rt_restir_di_spatial_enabled: inputs.rt_settings.restir_di_spatial_enabled,
rt_restir_di_spatial_sample_count: inputs.rt_settings.restir_di_spatial_sample_count,
rt_restir_gi_enabled: inputs.rt_settings.restir_gi_enabled,
rt_temporal_denoise_enabled: inputs.rt_settings.temporal_denoise_enabled,
```

In `src/render/runtime.rs`, pass RT settings into VPT inputs:

```rust
rt_settings: input.settings.rt,
```

- [ ] **Step 4: Run tests to verify GREEN**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'
cargo test --lib render::vpt_pipeline::tests::capture_render_mode_name_uses_manifest_values render::vpt_pipeline::tests::capture_rt_debug_view_name_uses_env_values render::runtime::tests::render_runtime_wires_capture_to_vpt_pipeline_only
```

Expected: PASS.

---

### Task 3: Visual Baseline Manifest And Script Contract

**Files:**
- Modify: `src/render/source_checks.rs`
- Modify: `run/visual-baselines.json`
- Modify: `run/validate-visual-baseline.ps1`

- [ ] **Step 1: Write failing source-contract tests**

In `visual_baseline_script_validates_captures_metadata_and_nonblank_ppm`, add tokens:

```rust
"REVOLUMETRIC_RENDER_MODE",
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
```

In `visual_baseline_manifest_covers_svgf_and_reblur_debug_cases`, add tokens:

```rust
"\"name\": \"rt_surface_debug\"",
"\"renderMode\": \"rt\"",
"\"expectedRenderBackend\": \"vpt\"",
"\"rtDebugView\": \"surface\"",
"\"rtRestirDi\": true",
"\"rtRestirDiSpatial\": true",
"\"rtRestirDiSpatialSamples\": 4",
"\"rtRestirGi\": true",
"\"rtTemporalDenoise\": true",
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'
cargo test --lib render::source_checks::tests::visual_baseline_script_validates_captures_metadata_and_nonblank_ppm render::source_checks::tests::visual_baseline_manifest_covers_svgf_and_reblur_debug_cases
```

Expected: FAIL because manifest/script do not contain RT contract fields.

- [ ] **Step 3: Add manifest RT case**

Append a case after `reblur_nrd_validation`:

```json
{
  "name": "rt_surface_debug",
  "renderMode": "rt",
  "denoiser": "svgf",
  "debugView": "final",
  "requiresNrd": false,
  "expectedEffectiveDenoiser": "svgf",
  "expectedRenderBackend": "vpt",
  "rtDebugView": "surface",
  "rtRestirDi": true,
  "rtRestirDiSpatial": true,
  "rtRestirDiSpatialSamples": 4,
  "rtRestirGi": true,
  "rtTemporalDenoise": true
}
```

- [ ] **Step 4: Update PowerShell env and metadata validation**

In `$previousEnv`, include the RT env vars listed in Task 3 Step 1.

In each case setup, set optional fields:

```powershell
if ($case.renderMode) { $env:REVOLUMETRIC_RENDER_MODE = $case.renderMode } else { Remove-Item Env:\REVOLUMETRIC_RENDER_MODE -ErrorAction SilentlyContinue }
if ($case.rtDebugView) { $env:REVOLUMETRIC_RT_DEBUG_VIEW = $case.rtDebugView } else { Remove-Item Env:\REVOLUMETRIC_RT_DEBUG_VIEW -ErrorAction SilentlyContinue }
if ($null -ne $case.rtRestirDi) { $env:REVOLUMETRIC_RT_RESTIR_DI = $case.rtRestirDi.ToString().ToLowerInvariant() } else { Remove-Item Env:\REVOLUMETRIC_RT_RESTIR_DI -ErrorAction SilentlyContinue }
if ($null -ne $case.rtRestirDiSpatial) { $env:REVOLUMETRIC_RT_RESTIR_DI_SPATIAL = $case.rtRestirDiSpatial.ToString().ToLowerInvariant() } else { Remove-Item Env:\REVOLUMETRIC_RT_RESTIR_DI_SPATIAL -ErrorAction SilentlyContinue }
if ($null -ne $case.rtRestirDiSpatialSamples) { $env:REVOLUMETRIC_RT_RESTIR_DI_SPATIAL_SAMPLES = "$($case.rtRestirDiSpatialSamples)" } else { Remove-Item Env:\REVOLUMETRIC_RT_RESTIR_DI_SPATIAL_SAMPLES -ErrorAction SilentlyContinue }
if ($null -ne $case.rtRestirGi) { $env:REVOLUMETRIC_RT_RESTIR_GI = $case.rtRestirGi.ToString().ToLowerInvariant() } else { Remove-Item Env:\REVOLUMETRIC_RT_RESTIR_GI -ErrorAction SilentlyContinue }
if ($null -ne $case.rtTemporalDenoise) { $env:REVOLUMETRIC_RT_TEMPORAL_DENOISE = $case.rtTemporalDenoise.ToString().ToLowerInvariant() } else { Remove-Item Env:\REVOLUMETRIC_RT_TEMPORAL_DENOISE -ErrorAction SilentlyContinue }
```

In `Assert-CaptureMetadata`, assert optional fields when present:

```powershell
if ($Case.expectedRenderBackend -and $Metadata.render_backend -ne $Case.expectedRenderBackend) {
    throw "capture metadata render_backend was $($Metadata.render_backend), expected $($Case.expectedRenderBackend)."
}
if ($Case.renderMode -and $Metadata.render_mode -ne $Case.renderMode) {
    throw "capture metadata render_mode was $($Metadata.render_mode), expected $($Case.renderMode)."
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
if ($null -ne $Case.rtTemporalDenoise -and [bool]$Metadata.rt_temporal_denoise_enabled -ne [bool]$Case.rtTemporalDenoise) {
    throw "capture metadata rt_temporal_denoise_enabled was $($Metadata.rt_temporal_denoise_enabled), expected $($Case.rtTemporalDenoise)."
}
```

- [ ] **Step 5: Run tests to verify GREEN**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'
cargo test --lib render::source_checks::tests::visual_baseline_script_validates_captures_metadata_and_nonblank_ppm render::source_checks::tests::visual_baseline_manifest_covers_svgf_and_reblur_debug_cases
```

Expected: PASS.

---

### Task 4: Full Verification And Commit

**Files:**
- Verify all changed files.

- [ ] **Step 1: Format**

Run:

```powershell
cargo fmt --check
```

Expected: PASS.

- [ ] **Step 2: Run library tests with shader compile skipped**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'
cargo test --lib
```

Expected: PASS.

- [ ] **Step 3: Run clippy**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'
cargo clippy --all-targets -- -D warnings
```

Expected: PASS.

- [ ] **Step 4: Run strict shader library tests**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='strict'
cargo test --lib
```

Expected: PASS.

- [ ] **Step 5: Check whitespace and status**

Run:

```powershell
git diff --check
git status --short
```

Expected: no whitespace errors and only intentional changed files before commit.

- [ ] **Step 6: Commit implementation**

Run:

```powershell
git add src/render/capture.rs src/render/vpt_pipeline.rs src/render/runtime.rs src/render/source_checks.rs run/visual-baselines.json run/validate-visual-baseline.ps1
git commit -m "feat: add RT visual baseline metadata contract"
```

Expected: commit succeeds.
