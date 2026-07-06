# RT Frame Status Design

## Goal

Expose the RT pipeline's last-frame readiness state through `RenderRuntimeStatus` and show it in the editor so operators can distinguish requested RT settings from the passes that actually produced a frame.

## Current Facts

- `RenderRuntimeStatus` currently reports only `actual_backend` and `rt_supported`.
- `RtRuntimePipeline` already tracks per-frame readiness in `RtPipelineFrameState`: surface, ReSTIR-DI history, ReSTIR-GI history, direct lighting, temporal, and resolve.
- The editor already receives `Option<RenderRuntimeStatus>` and displays requested RT toggles, actual backend, and RT support.
- RT ReSTIR-DI/GI toggles are requested settings; they do not prove the optional RT passes rendered on the last RT frame.
- The project already uses source-contract tests where constructing a live Vulkan runtime is unsuitable for unit tests.

## Design

Add a small copyable `RtFrameStatus` snapshot in `src/render/rt_pipeline.rs`:

- `frame_resources_ready`
- `surface_ready`
- `restir_di_history_ready`
- `restir_gi_history_ready`
- `direct_lighting_ready`
- `temporal_ready`
- `resolve_ready`

Expose it with `RtRuntimePipeline::frame_status()`. The method must read the pipeline's owned state only; it must not inspect UI settings or infer pass activity from requested toggles.

Extend `RenderRuntimeStatus` with:

- `rt_frame_status: Option<RtFrameStatus>`

`RenderRuntime::status()` should include a snapshot when RT is the actual backend or when RT frame resources already exist. It should return `None` when the runtime has no meaningful RT frame state to show, such as VPT-only fallback before any RT resources were created.

Update the editor to display the snapshot in compact operational labels:

- top bar: a single `rt_frame` state label
- render panel: a short readiness row for frame, surface, direct lighting, temporal, resolve, DI history, and GI history
- console: machine-readable tokens for the same state

The UI should show `pending` when runtime status is absent, `inactive` when there is a runtime but no RT frame status, `warming` when RT resources exist but resolve is not ready, and `ready` once the resolve output was produced by the RT pipeline.

## Non-Goals

- Do not change shader code.
- Do not change RT pass ordering.
- Do not add a new editor panel.
- Do not add GPU-dependent unit tests.
- Do not claim perceptual visual correctness from these status fields.

## Testing Strategy

Use TDD with source-contract and helper tests:

- RT pipeline test proves `RtFrameStatus` exists and `frame_status()` maps each `RtPipelineFrameState` field into the public snapshot.
- Runtime source test proves `RenderRuntimeStatus` carries `Option<RtFrameStatus>` and `RenderRuntime::status()` uses `RtRuntimePipeline::frame_status()`.
- Editor helper tests prove `pending`, `inactive`, `warming`, and `ready` labels.
- Editor source tests prove the top bar, render panel, and console expose RT frame readiness.

Run the existing local verification set after implementation:

- `cargo fmt --check`
- `REVOLUMETRIC_SHADER_COMPILE=skip cargo test --lib`
- `REVOLUMETRIC_SHADER_COMPILE=skip cargo clippy --all-targets -- -D warnings`
- `REVOLUMETRIC_SHADER_COMPILE=strict cargo test --lib`
- `git diff --check`

## Risks

The snapshot describes the last recorded RT frame, not the frame currently being built by the GPU. This is acceptable for editor observability because it avoids synchronizing UI code with in-flight command buffers. Requested RT toggles remain visible separately, so users can compare requested settings with active pass history.
