# RT Runtime Status And Backend Switching Design

## Goal

Make the editor display the actual renderer backend selected by `RenderRuntime`, and make runtime backend selection respond to live UI changes of `LightingSettings.render_mode`.

## Current Facts

- The editor can now mutate `LightingSettings.render_mode` and `RtSettings`.
- `RenderRuntime::new` resolves `RenderBackend` from the initial requested render mode and probed RT capabilities.
- `RenderRuntime::render_frame` and resize paths currently continue using the stored backend after initialization.
- `RenderRuntime` already owns the authoritative `RenderBackend` and `RtCapabilities`.
- The editor has no read-only runtime status input, so it can only display requested settings.

## Design

Add a small read-only `RenderRuntimeStatus` snapshot in `src/render/runtime.rs`:

- `actual_backend: RenderBackend`
- `rt_supported: bool`

Expose it through `RenderRuntime::status()`. This keeps runtime authority in the renderer and gives the editor enough information to distinguish requested mode from the actual backend.

Add a private runtime helper that refreshes `self.render_backend` from the current frame or resize settings:

- call it at the start of `render_frame` after `begin_frame`
- call it at the start of `resize_pipeline_to_swapchain`
- preserve the existing backend-change history reset path
- warn when explicit `RenderMode::Rt` resolves to `RenderBackend::Vpt`

Pass `Option<RenderRuntimeStatus>` from `RevolumetricApp::build_egui_frame` into `EditorUiFrameState`. The option keeps the editor usable in tests and during any future pre-runtime frame. The UI should display:

- requested render mode from live `LightingSettings`
- actual backend from `RenderRuntimeStatus`
- RT support boolean from `RenderRuntimeStatus`

## Non-Goals

- Do not change ray tracing shader code.
- Do not change RT pass ordering.
- Do not add a new editor panel.
- Do not make editor UI own or cache render backend state.
- Do not add GPU-dependent integration tests.

## Testing Strategy

Use source-contract tests for runtime/app/editor wiring because live Vulkan runtime construction is not suitable for unit tests. Add helper tests for backend/status labels.

Required checks:

- runtime source test proving backend refresh happens before pass selection in `render_frame`
- runtime source test proving resize refreshes backend from current settings
- runtime source test proving `RenderRuntimeStatus` and `RenderRuntime::status()` expose backend and RT support
- app source test proving runtime status is passed into `EditorUiFrameState`
- editor source tests proving top bar and console display actual backend and RT support
- helper test for backend/status labels

## Risks

The editor status is a snapshot captured before the current egui frame mutates settings, so a newly changed renderer selection becomes visible as the actual backend on the next frame after runtime consumes it. This is acceptable for a read-only status surface and avoids introducing a two-way UI/runtime event system.
