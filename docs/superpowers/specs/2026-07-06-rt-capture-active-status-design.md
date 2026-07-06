# RT Capture Active Status Design

## Goal

Record actual RT pass activity in capture metadata and validate it in the RT visual baseline case. This lets a capture prove not only which RT settings were requested, but also whether the RT graph and optional ReSTIR passes participated in the captured frame.

## Current Facts

- `CaptureMetadata` records requested RT settings such as `rt_restir_di_enabled`, `rt_restir_gi_enabled`, and `rt_temporal_denoise_enabled`.
- `RtRuntimePipeline::record_and_execute_frame` already tracks actual local frame activity through `rt_graph_rendered`, `rt_restir_di_rendered`, and `rt_restir_gi_rendered`.
- VPT fallback captures can carry requested RT settings when `RenderMode::Rt` falls back to VPT, so requested RT settings are not enough to prove RT execution.
- The visual baseline script already validates capture metadata fields from `run/visual-baselines.json`.

## Design

Extend `CaptureMetadata` with four actual RT activity fields:

- `rt_frame_rendered`: true when the RT graph produced the resolve output for this frame.
- `rt_restir_di_rendered`: true when RT ReSTIR-DI produced reservoir resources for this frame.
- `rt_restir_gi_rendered`: true when RT ReSTIR-GI produced reservoir resources for this frame.
- `rt_resolve_ready`: true when the RT resolve output was queued for capture.

For RT captures, populate these fields from `RtRuntimePipeline` local frame state:

- `rt_frame_rendered` is true in the capture path because the RT graph path has reached resolve registration.
- `rt_restir_di_rendered` uses the already-computed `rt_restir_di_rendered` local.
- `rt_restir_gi_rendered` uses the already-computed `rt_restir_gi_rendered` local.
- `rt_resolve_ready` is true in the RT resolve capture path.

For VPT captures, set all four fields to false. This makes fallback metadata explicit and keeps a stable schema for every capture.

Extend the RT visual baseline manifest with expected active RT fields, and update `run/validate-visual-baseline.ps1` to assert them when present. VPT cases should not need to spell out false values unless a future case wants to assert fallback behavior.

## Non-Goals

- Do not change shader code.
- Do not change RT pass ordering.
- Do not add perceptual image comparison.
- Do not require RT-capable hardware for the default visual baseline run.

## Testing Strategy

Use TDD:

- Extend `metadata_json_records_frame_settings_and_paths` so the test fails until `CaptureMetadata` serializes the new active RT fields.
- Add RT pipeline source-contract coverage proving the RT capture path writes active fields from `rt_graph_rendered`, `rt_restir_di_rendered`, and `rt_restir_gi_rendered`.
- Add VPT pipeline source-contract coverage proving VPT capture writes all active RT fields as false.
- Extend source checks for the visual baseline script and manifest.
- Run `run/validate-visual-baseline.ps1 -Rt` after implementation to confirm the new metadata fields are produced and validated on the local RT path.

## Risks

These fields are capture-time facts, not a full frame timeline. They should not be used as a performance profiler or as proof of visual quality. They only prove that the relevant RT graph resources were registered for the captured frame.
