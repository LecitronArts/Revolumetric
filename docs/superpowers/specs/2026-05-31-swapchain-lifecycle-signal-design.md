# Swapchain Lifecycle Signal Design

## Goal

Make swapchain recreation observable to `RenderRuntime` so VPT resources can be resized after every swapchain lifecycle change, not only after `WindowEvent::Resized`.

## Current Facts

- `RenderDevice::handle_resize` explicitly calls `recreate_swapchain`.
- `RenderDevice::begin_frame` also calls `recreate_swapchain` when image acquire returns `ERROR_OUT_OF_DATE_KHR`, then returns `FrameContext::skip`.
- `RenderDevice::end_frame` calls `recreate_swapchain` when present is suboptimal or out of date.
- `RenderRuntime::resize` already handles explicit window resize by calling `RenderDevice::handle_resize`, ensuring passes, and resizing `VptRuntimePipeline`.
- `RenderRuntime::render_frame` currently receives no signal when `begin_frame` or `end_frame` recreates the swapchain internally.

## Design

Add a small value type in `src/render/frame.rs`:

```rust
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FrameCompletion {
    pub swapchain_recreated: bool,
}
```

Extend `FrameContext` with:

```rust
pub swapchain_recreated: bool,
```

`FrameContext::skip_after_swapchain_recreate(frame_index)` should return a skipped frame with `swapchain_recreated = true`. Existing `FrameContext::skip(frame_index)` remains available for skipped frames that do not recreate the swapchain.

Change `RenderDevice::end_frame` from `Result<()>` to `Result<FrameCompletion>`. It returns `swapchain_recreated = true` when present handling recreates the swapchain, and `false` otherwise.

Add a private runtime helper:

```rust
fn resize_pipeline_to_swapchain(&mut self, ucvh: Option<&Ucvh>, settings: RuntimeSettings, restir_di_enabled: bool, area_restir_enabled: bool) -> Result<()>
```

This helper should use `self.renderer.swapchain_extent()` after recreation and then ensure/resize the VPT pipeline. `RenderRuntime::resize` should call `handle_resize` and then this helper. `RenderRuntime::render_frame` should call the helper when:

- `begin_frame` returns a skipped frame with `swapchain_recreated = true`.
- `end_frame` returns `FrameCompletion { swapchain_recreated: true }`.
- The no-scene-UBO path still calls `end_frame`; if completion reports recreation, resize synchronization still runs.

## Non-Goals

- Do not change swapchain creation policy, present mode selection, or frame acquisition order.
- Do not change VPT pass resize semantics.
- Do not add transient resource allocation or descriptor automation in this phase.
- Do not make Vulkan objects directly unit-testable.

## Testing

Use source-level tests and pure value tests:

- `FrameContext::skip_after_swapchain_recreate` sets `should_render = false` and `swapchain_recreated = true`.
- `FrameCompletion::default()` reports `swapchain_recreated = false`.
- `RenderDevice::end_frame` returns `Result<FrameCompletion>` and contains a true assignment on present-driven recreate.
- `RenderRuntime::render_frame` checks `frame.swapchain_recreated` and `completion.swapchain_recreated`.
- `RenderRuntime::resize` delegates pipeline resize through the same helper used by internal recreate paths.

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib render::frame render::runtime::tests::render_runtime_observes_swapchain_recreate_signals render::device::tests::device_reports_swapchain_recreation_from_frame_completion; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
```

## Acceptance Criteria

- Swapchain recreation is represented in frame-level return data.
- Runtime pass resize synchronization is reused by explicit window resize and internal device recreation paths.
- Existing frame preparation ordering tests still pass.
- Full library tests pass with shader compilation skipped.

