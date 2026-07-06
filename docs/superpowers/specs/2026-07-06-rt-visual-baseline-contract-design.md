# RT Visual Baseline Contract Design

## Goal

Extend the visual baseline contract so captures can identify the requested render mode, actual backend, and RT controls. This makes automated capture output distinguish VPT fallback from hardware RT and records the RT feature state that produced a baseline.

The current implementation only captures the VPT postprocess image. Hardware RT frames still return `pending_capture: None`, so this design does not claim full RT image readback support. It prepares metadata, manifest, script validation, and source-contract tests for the next RT readback step.

## Current Facts

- `src/render/capture.rs` writes metadata for frame index, dimensions, paths, VPT ReSTIR state, VPT debug view, and denoiser state.
- `src/render/vpt_pipeline.rs` copies the VPT postprocess image into `RenderCapture` and creates `CaptureMetadata`.
- `src/render/rt_pipeline.rs` blits RT resolve output to the swapchain but returns `pending_capture: None`.
- `run/visual-baselines.json` contains only VPT/NRD cases.
- `run/validate-visual-baseline.ps1` sets and checks VPT capture env only. It does not set or validate `REVOLUMETRIC_RENDER_MODE` or RT settings.

## Approach

Use an explicit capture metadata contract rather than inferring backend state from case names or output paths.

`CaptureMetadata` gains these fields:

- `render_backend`: actual backend that produced the captured image, initially `vpt` for VPT capture output.
- `render_mode`: requested render mode from `LightingSettings`, such as `auto`, `vpt`, or `rt`.
- `rt_debug_view`: RT debug view name from `RtSettings`.
- `rt_restir_di_enabled`
- `rt_restir_di_spatial_enabled`
- `rt_restir_di_spatial_sample_count`
- `rt_restir_gi_enabled`
- `rt_temporal_denoise_enabled`

The VPT pipeline receives `RtSettings` in `VptFrameInputs` so VPT fallback captures can record the RT controls that were requested alongside the actual backend. This matters when `REVOLUMETRIC_RENDER_MODE=rt` falls back to VPT on unsupported hardware.

## Manifest And Script Contract

`run/visual-baselines.json` gains an RT-oriented case named `rt_surface_debug` with:

- `renderMode`: `rt`
- `expectedRenderBackend`: `vpt`
- `rtDebugView`: `surface`
- `rtRestirDi`: `true`
- `rtRestirDiSpatial`: `true`
- `rtRestirDiSpatialSamples`: `4`
- `rtRestirGi`: `true`
- `rtTemporalDenoise`: `true`

The first RT contract case intentionally expects `vpt` as the captured backend because the current readback path is VPT-only. This records the fallback behavior without requiring RT hardware on CI. A future RT readback task can add a hardware-gated case with `expectedRenderBackend: "rt"` after `RtRuntimePipeline` emits capture metadata and copies the RT resolve image to readback.

`run/validate-visual-baseline.ps1` will preserve, set, and restore the relevant RT env vars:

- `REVOLUMETRIC_RENDER_MODE`
- `REVOLUMETRIC_RT_DEBUG_VIEW`
- `REVOLUMETRIC_RT_RESTIR_DI`
- `REVOLUMETRIC_RT_RESTIR_DI_SPATIAL`
- `REVOLUMETRIC_RT_RESTIR_DI_SPATIAL_SAMPLES`
- `REVOLUMETRIC_RT_RESTIR_GI`
- `REVOLUMETRIC_RT_TEMPORAL_DENOISE`

The script validates optional manifest fields only when present, so existing VPT/NRD cases keep working.

## Tests

Tests are source-contract and serialization tests because full GPU capture execution depends on runtime hardware and shader state.

- `capture.rs` verifies `CaptureMetadata::to_json()` emits all backend and RT fields.
- `source_checks.rs` verifies the visual baseline manifest contains the RT contract case.
- `source_checks.rs` verifies the validation script stores/restores RT env vars, sets them from manifest case fields, and checks metadata fields.
- `runtime.rs` or `vpt_pipeline.rs` source-contract tests verify `RuntimeSettings.rt` is threaded into `VptFrameInputs`.

## Boundaries

This design does not implement RT resolve readback. The required follow-up is to add `capture: Option<&mut RenderCapture>` to `RtFrameInputs`, copy `rt_resolve_outputs.output` into the readback buffer after resolve, and return `CaptureMetadata` with `render_backend: "rt"`. That work needs graph dependency review and should be tested separately.

## Approval

The user granted `TRUSTED` autonomy for project RT pipeline completion. This design is approved for implementation under that authorization.
