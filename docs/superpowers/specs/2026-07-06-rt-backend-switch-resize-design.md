# RT Backend Switch Resize Safety Design

## Goal

Keep RT and VPT GPU frame resources dimensionally valid when the user switches renderer backend after a window or swapchain resize.

## Current Facts

- `RenderRuntime` can now refresh `render_backend` from live `LightingSettings.render_mode`.
- `resize_pipeline_to_swapchain` currently resizes only the currently selected backend.
- `ensure_passes` creates missing passes at the current swapchain extent, but it does not resize existing passes.
- A backend that was active earlier can keep old-size pass images or buffers while inactive.
- Switching back to that backend after a resize can therefore reuse stale-size resources.

## Design

Keep all GPU resource replacement in the resize path. Do not resize from inside frame recording after `begin_frame`.

On every runtime resize or swapchain recreation:

- refresh the selected backend from current settings
- ensure passes for the selected backend, matching the existing runtime pattern
- resize the selected backend
- also resize the non-selected backend when it already owns frame resources

Add small pipeline helpers:

- `RtRuntimePipeline::has_frame_resources()`
- `VptRuntimePipeline::has_frame_resources()`

Add runtime helper methods that decide whether each backend needs resizing from either current selection or pre-existing frame resources.

## Non-Goals

- Do not change RT shader code.
- Do not change render graph pass ordering.
- Do not create missing passes for the inactive backend.
- Do not resize GPU resources while a frame is actively being recorded.

## Testing Strategy

Use source-contract tests because live Vulkan resize/backend-switch scenarios are not suitable for unit tests.

Required tests:

- RT pipeline exposes a frame-resource presence helper.
- VPT pipeline exposes a frame-resource presence helper.
- `resize_pipeline_to_swapchain` routes resize through RT/VPT helper methods instead of directly matching only the selected backend.
- Runtime resize helpers resize the backend when it is selected or when it already owns frame resources.
- `render_frame` must not call resize helpers purely because the backend changed; resize stays in swapchain resize handling.

## Risk

Inactive VPT resize still requires `UcvhGpuResources`, matching the existing VPT resize API. If no UCVH GPU resources exist, VPT cannot have fully usable VPT frame resources, so skipping that resize remains consistent with current behavior.
