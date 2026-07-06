# RT Frame Skip Reason Design

## Goal

Expose the last RT frame's primary skip reason through the existing runtime status path and show it in the editor, so an operator can distinguish a warming RT pipeline from a frame that fell back because required RT inputs or passes were missing.

## Current Facts

- `RtFrameStatus` currently exposes only readiness booleans: frame resources, surface, ReSTIR-DI history, ReSTIR-GI history, direct lighting, temporal, and resolve.
- `RtRuntimePipeline::record_and_execute_frame` already has distinct fallback branches for UCVH upload readiness, missing CPU UCVH scene, missing acceleration structure loader, failed AS rebuild, missing TLAS/AABB resources, missing GPU UCVH descriptors, and required RT pass initialization.
- The editor already receives `Option<RenderRuntimeStatus>` and displays RT frame readiness in the top bar, render panel, and console.
- Existing backend fallback reporting is separate: unsupported RT hardware or requested RT with VPT active is already described by `rt_backend_notice`.

## Design

Add a copyable enum in `src/render/rt_pipeline.rs`:

```rust
pub enum RtFrameSkipReason {
    UcvhUploadPending,
    CpuUcvhSceneMissing,
    AccelerationStructureLoaderMissing,
    AccelerationStructureRebuildFailed,
    AccelerationStructureMissing,
    UcvhGpuDescriptorsMissing,
    RequiredPassesMissing,
}
```

Extend `RtFrameStatus` and `RtPipelineFrameState` with:

```rust
pub skip_reason: Option<RtFrameSkipReason>
```

`RtRuntimePipeline::record_and_execute_frame` should set the most specific reason available before adding fallback clear/present work. Earlier root causes must win over downstream symptoms: for example, when `ucvh_ready` is false, the user should see `ucvh_upload_pending`, not a later missing TLAS/AABB symptom. When the RT graph reaches resolve registration and `rt_graph_rendered` becomes true, clear the skip reason.

Use this priority:

1. `UcvhUploadPending` when the runtime has not uploaded UCVH data yet.
2. `CpuUcvhSceneMissing` when RT is active but no CPU UCVH scene is available for AS rebuild.
3. `AccelerationStructureLoaderMissing` when Vulkan RT support should exist but the AS loader is absent.
4. `AccelerationStructureRebuildFailed` when GPU AS rebuild returns an error.
5. `RequiredPassesMissing` when surface, direct lighting, temporal, or resolve pass resources are missing.
6. `AccelerationStructureMissing` when TLAS or AABB input resources are absent after AS rebuild handling.
7. `UcvhGpuDescriptorsMissing` when TLAS/AABB exist but GPU UCVH descriptors are absent.

Optional RT ReSTIR-DI/GI pass misses are not frame skip reasons in this slice. They can degrade lighting features but the RT frame can still produce a resolve output, and actual optional-pass activity is already recorded separately in capture metadata.

Update `src/editor/ui.rs` with a helper:

```rust
fn rt_frame_skip_reason_label(status: Option<RenderRuntimeStatus>) -> &'static str
```

Labels should be stable short tokens for console and compact UI:

- `pending` when runtime status is absent.
- `inactive` when runtime status exists but no RT frame status exists.
- `none` when RT frame status exists and has no skip reason.
- `ucvh_upload_pending`, `cpu_ucvh_missing`, `as_loader_missing`, `as_rebuild_failed`, `as_missing`, `ucvh_gpu_missing`, `required_passes_missing` for enum variants.

Display the label in:

- top bar as `rt_reason <token>`
- render panel as `RT reason`
- console as `rt_skip_reason=<token>`

## Non-Goals

- Do not change shader code or RT pass ordering.
- Do not add scrolling console log events for every skip; repeated per-frame messages would spam the editor.
- Do not merge backend unsupported/fallback notices into this field.
- Do not claim visual quality from the reason field.
- Do not treat optional RT ReSTIR-DI/GI pass absence as a full-frame skip reason.

## Testing Strategy

Use TDD with focused helper/source tests:

- RT pipeline tests prove `RtFrameStatus` includes `skip_reason`, defaults to `None`, snapshots `RtPipelineFrameState`, and source-contract checks cover every fallback branch assigning a structured reason.
- Runtime source tests prove `RenderRuntimeStatus` still carries `Option<RtFrameStatus>` and uses `RtRuntimePipeline::frame_status()`.
- Editor helper tests cover `pending`, `inactive`, `none`, and every skip reason label.
- Editor source tests prove the top bar, render panel, and console display the skip reason helper.

Run the existing verification set after implementation:

- `cargo fmt --check`
- `REVOLUMETRIC_SHADER_COMPILE=skip cargo test --lib`
- `REVOLUMETRIC_SHADER_COMPILE=skip cargo clippy --all-targets -- -D warnings`
- `REVOLUMETRIC_SHADER_COMPILE=strict cargo test --lib`
- `git diff --check`

## Risks

The editor status remains a last-frame snapshot and can lag by one frame because UI frame state is built before recording the next render frame. That matches the current runtime-status architecture and avoids synchronizing UI code with in-flight GPU work. Source-contract checks cannot prove live Vulkan branch execution, so they should be paired with the existing RT visual baseline run when hardware validation is needed.
