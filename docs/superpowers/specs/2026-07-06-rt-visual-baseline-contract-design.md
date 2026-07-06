# RT Visual Baseline Contract Design

## Goal

Extend the visual baseline contract so captures can identify the requested render mode, actual backend, and RT controls. Add RT resolve readback so hardware RT frames can write the same PPM/JSON capture artifacts as the VPT postprocess path.

## Starting Facts Before This Slice

- `src/render/capture.rs` writes metadata for frame index, dimensions, paths, VPT ReSTIR state, VPT debug view, and denoiser state.
- `src/render/vpt_pipeline.rs` copies the VPT postprocess image into `RenderCapture` and creates `CaptureMetadata`.
- `src/render/rt_pipeline.rs` blits RT resolve output to the swapchain but returns `pending_capture: None`.
- `run/visual-baselines.json` contains only VPT/NRD cases.
- `run/validate-visual-baseline.ps1` sets and checks VPT capture env only. It does not set or validate `REVOLUMETRIC_RENDER_MODE` or RT settings.

## Approach

Use an explicit capture metadata contract rather than inferring backend state from case names or output paths.

`CaptureMetadata` gains these fields:

- `render_backend`: actual backend that produced the captured image, `vpt` for VPT postprocess output and `rt` for RT resolve output.
- `render_mode`: requested render mode from `LightingSettings`, such as `auto`, `vpt`, or `rt`.
- `rt_debug_view`: RT debug view name from `RtSettings`.
- `rt_restir_di_enabled`
- `rt_restir_di_spatial_enabled`
- `rt_restir_di_spatial_sample_count`
- `rt_restir_gi_enabled`
- `rt_temporal_denoise_enabled`

The VPT pipeline receives `RtSettings` in `VptFrameInputs` so VPT fallback captures can record the RT controls that were requested alongside the actual backend.

The RT pipeline receives `RenderCapture` in `RtFrameInputs`. After `rt_resolve.register_graph`, it registers a `capture_rt_resolve` transfer pass that copies `rt_resolve_outputs.output` into the capture readback buffer, then returns `CaptureMetadata` with `source: "rt_resolve_output"` and `render_backend: "rt"`. The blit pass depends on the capture pass when capture is queued, matching the ordering used by the VPT postprocess capture path.

## Manifest And Script Contract

`run/visual-baselines.json` pins the existing VPT/NRD cases to `renderMode: "vpt"` so default `auto` cannot route them to RT on RT-capable machines.

The manifest also gains a hardware-gated RT case named `rt_surface_debug` with:

- `renderMode`: `rt`
- `requiresRt`: `true`
- `expectedRenderBackend`: `rt`
- `rtDebugView`: `surface`
- `rtRestirDi`: `true`
- `rtRestirDiSpatial`: `true`
- `rtRestirDiSpatialSamples`: `4`
- `rtRestirGi`: `true`
- `rtTemporalDenoise`: `true`

`run/validate-visual-baseline.ps1` will preserve, set, and restore the relevant RT env vars:

- `REVOLUMETRIC_RENDER_MODE`
- `REVOLUMETRIC_RT_DEBUG_VIEW`
- `REVOLUMETRIC_RT_RESTIR_DI`
- `REVOLUMETRIC_RT_RESTIR_DI_SPATIAL`
- `REVOLUMETRIC_RT_RESTIR_DI_SPATIAL_SAMPLES`
- `REVOLUMETRIC_RT_RESTIR_GI`
- `REVOLUMETRIC_RT_TEMPORAL_DENOISE`

The script validates optional manifest fields only when present. It skips `requiresRt` cases unless invoked with `-Rt`, keeping the default local baseline path stable on machines without RT hardware while allowing explicit RT capture validation where supported.

## Tests

Tests are source-contract and serialization tests because full GPU capture execution depends on runtime hardware and shader state.

- `capture.rs` verifies `CaptureMetadata::to_json()` emits all backend and RT fields.
- `source_checks.rs` verifies the visual baseline manifest pins VPT cases and contains the hardware-gated RT contract case.
- `source_checks.rs` verifies the validation script stores/restores RT env vars, sets them from manifest case fields, gates RT cases behind `-Rt`, and checks metadata fields.
- `vpt_pipeline.rs` source-contract/helper tests verify `RuntimeSettings.rt` is threaded into `VptFrameInputs` and serialized with stable manifest values.
- `rt_pipeline.rs` source-contract tests verify `RtFrameInputs` carries capture, `capture_rt_resolve` copies resolve output, and `pending_capture` returns RT metadata.
- `runtime.rs` source-contract tests verify `RenderRuntime` passes capture into both RT and VPT pipeline inputs.

## Boundaries

This design adds capture readback for RT resolve output. It does not add perceptual image comparison, golden-image diff thresholds, or automatic hardware detection. The RT visual baseline case is explicit and hardware-gated by `-Rt`; without that flag, the script skips it.

## Approval

The user granted `TRUSTED` autonomy for project RT pipeline completion. This design is approved for implementation under that authorization.
