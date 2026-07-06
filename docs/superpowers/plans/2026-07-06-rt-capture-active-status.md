# RT Capture Active Status Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add actual RT pass activity fields to capture metadata and validate them in the RT visual baseline case.

**Architecture:** `CaptureMetadata` owns the schema. RT and VPT runtime pipelines populate the schema from their own frame facts, while `run/validate-visual-baseline.ps1` validates optional expected fields from `run/visual-baselines.json`.

**Tech Stack:** Rust 2024, PowerShell visual baseline wrapper, existing source-contract tests, `cargo test --lib`, `cargo clippy`, strict shader compilation.

---

## File Structure

- Modify `src/render/capture.rs`: add active RT fields to `CaptureMetadata`, JSON serialization, and metadata JSON tests.
- Modify `src/render/rt_pipeline.rs`: populate active RT fields from `rt_graph_rendered`, `rt_restir_di_rendered`, `rt_restir_gi_rendered`, and add source-contract coverage.
- Modify `src/render/vpt_pipeline.rs`: populate active RT fields as false for VPT captures and add source-contract coverage.
- Modify `run/visual-baselines.json`: add expected active RT fields to the `rt_surface_debug` case.
- Modify `run/validate-visual-baseline.ps1`: assert optional active RT fields.
- Modify `src/render/source_checks.rs`: pin the script and manifest tokens.
- Modify `README.md` and `run/README.md`: mention that RT capture metadata records requested and active RT state.

## Task 1: Metadata Schema

- [ ] **Step 1: Write failing capture metadata test**

In `src/render/capture.rs`, extend `metadata_json_records_frame_settings_and_paths` with struct fields:

```rust
rt_frame_rendered: true,
rt_restir_di_rendered: true,
rt_restir_gi_rendered: false,
rt_resolve_ready: true,
```

Add assertions:

```rust
assert!(json.contains("\"rt_frame_rendered\": true"));
assert!(json.contains("\"rt_restir_di_rendered\": true"));
assert!(json.contains("\"rt_restir_gi_rendered\": false"));
assert!(json.contains("\"rt_resolve_ready\": true"));
```

- [ ] **Step 2: Run RED**

Run:

```powershell
REVOLUMETRIC_SHADER_COMPILE=skip cargo test metadata_json_records_frame_settings_and_paths --lib
```

Expected: compile failure because the new `CaptureMetadata` fields do not exist.

- [ ] **Step 3: Implement metadata fields and JSON**

Add the four boolean fields to `CaptureMetadata` after `rt_temporal_denoise_enabled`, serialize them in `to_json`, and pass the new values into `format!`.

- [ ] **Step 4: Run GREEN**

Run the same focused test and expect it to pass.

## Task 2: RT/VPT Pipeline Population

- [ ] **Step 1: Write failing source-contract tests**

In `src/render/rt_pipeline.rs`, add a test checking the RT capture path contains:

```rust
"rt_frame_rendered:rt_graph_rendered"
"rt_restir_di_rendered"
"rt_restir_gi_rendered"
"rt_resolve_ready:true"
```

In `src/render/vpt_pipeline.rs`, add a test checking VPT capture metadata contains:

```rust
"rt_frame_rendered:false"
"rt_restir_di_rendered:false"
"rt_restir_gi_rendered:false"
"rt_resolve_ready:false"
```

- [ ] **Step 2: Run RED**

Run:

```powershell
REVOLUMETRIC_SHADER_COMPILE=skip cargo test rt_pipeline_capture_metadata_records_active_rt_passes vpt_capture_metadata_marks_active_rt_passes_false --lib
```

Expected: failures because the new metadata fields are not populated.

- [ ] **Step 3: Populate metadata**

In `src/render/rt_pipeline.rs`, set active RT metadata in the `CaptureMetadata` literal:

```rust
rt_frame_rendered: rt_graph_rendered,
rt_restir_di_rendered,
rt_restir_gi_rendered,
rt_resolve_ready: true,
```

In `src/render/vpt_pipeline.rs`, set all four fields to false.

- [ ] **Step 4: Run GREEN**

Run the focused pipeline tests and expect them to pass.

## Task 3: Visual Baseline Contract

- [ ] **Step 1: Write failing source checks**

In `src/render/source_checks.rs`, extend visual baseline script/manifest checks for:

```rust
"rt_frame_rendered"
"rt_restir_di_rendered"
"rt_restir_gi_rendered"
"rt_resolve_ready"
"\"expectedRtFrameRendered\": true"
"\"expectedRtRestirDiRendered\": true"
"\"expectedRtRestirGiRendered\": true"
"\"expectedRtResolveReady\": true"
```

- [ ] **Step 2: Run RED**

Run:

```powershell
REVOLUMETRIC_SHADER_COMPILE=skip cargo test visual_baseline_script_validates_captures_metadata_and_nonblank_ppm visual_baseline_manifest_covers_svgf_and_reblur_debug_cases --lib
```

Expected: failures because manifest and script do not yet include these tokens.

- [ ] **Step 3: Update manifest and script**

Add expected fields to the `rt_surface_debug` manifest case:

```json
"expectedRtFrameRendered": true,
"expectedRtRestirDiRendered": true,
"expectedRtRestirGiRendered": true,
"expectedRtResolveReady": true
```

In `Assert-CaptureMetadata`, add optional boolean assertions for those properties.

- [ ] **Step 4: Update docs**

Update `README.md` and `run/README.md` so the visual baseline section says capture metadata records requested RT controls and active RT frame/pass state.

- [ ] **Step 5: Run GREEN**

Run the focused source-check tests and expect them to pass.

## Task 4: Verification And Commit

- [ ] Run `cargo fmt --check`.
- [ ] Run `REVOLUMETRIC_SHADER_COMPILE=skip cargo test --lib`.
- [ ] Run `REVOLUMETRIC_SHADER_COMPILE=skip cargo clippy --all-targets -- -D warnings`.
- [ ] Run `REVOLUMETRIC_SHADER_COMPILE=strict cargo test --lib`.
- [ ] Run `.\run\validate-visual-baseline.ps1 -Rt`.
- [ ] Run `git diff --check`.
- [ ] Run `git status --short --branch`.
- [ ] Commit with message `feat: record active RT capture status`.
