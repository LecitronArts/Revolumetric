# RT UI Integration Design

## Goal

Expose the existing hardware RT runtime controls in the egui editor shell, then cover the UI-to-runtime setting flow with focused tests.

This is a UI and settings integration phase. It does not change RT acceleration-structure construction, ray tracing shader algorithms, ReSTIR reservoir math, or denoising behavior.

## Current Facts

- `RevolumetricApp` already owns `RtSettings` and passes it into `RenderRuntime` through `RuntimeSettings`.
- `RenderRuntime` routes frames to `RtRuntimePipeline` or `VptRuntimePipeline` from `LightingSettings.render_mode` and probed RT capability support.
- `RtSettings` already controls RT ReSTIR-DI, RT ReSTIR-DI spatial reuse, RT ReSTIR-GI, RT temporal accumulation, temporal thresholds, history length, and `RtDebugView`.
- The editor currently receives mutable `LightingSettings`, `RestirDiSettings`, and `AreaRestirSettings`, but it does not receive mutable `RtSettings`.
- The existing editor pattern is direct mutation of runtime settings inside `show_render_panel`, `show_restir_panel`, and `show_debug_panel`.
- Baseline validation in the isolated worktree passed with `REVOLUMETRIC_SHADER_COMPILE=skip cargo test --lib` and 673 passing tests.

## Recommended Approach

Extend the existing editor panels instead of creating a new RT-only panel.

Reasoning:

- It matches the current UI structure and avoids a new navigation branch for a small first phase.
- It uses the same settings mutation model already used for VPT, ReSTIR-DI, and Area ReSTIR.
- It keeps this phase limited to UI and tests, so RT backend behavior remains easier to reason about.

Rejected alternatives:

- A new `EditorPanel::Rt` would make RT controls more isolated, but it would split renderer controls across more panels without adding runtime capability feedback yet.
- A richer RT status UI with actual backend and capability readback is useful, but it needs a new read-only runtime-to-editor state path. That belongs in a later phase after this direct setting path is covered.

## UI Design

### Top Bar And Console

The top bar should stop presenting the editor as VPT-only. It should show the requested renderer mode and keep the existing denoiser combo.

The console summary should include:

- requested render mode
- RT ReSTIR-DI enabled state
- RT ReSTIR-GI enabled state
- RT temporal denoise enabled state

This is request-state display, not actual backend readback. Actual backend readback is out of scope for this phase.

### Render Panel

Add a "Renderer" group to the existing Render panel:

- render mode combo: `Auto`, `VPT`, `RT`
- exposure remains in the shared render/display section

Keep the existing VPT controls:

- max bounces
- denoiser mode
- fallback A-trous iterations

Add an "RT Temporal" group:

- temporal denoise toggle
- history length slider, clamped to `1..=64`
- normal threshold slider, clamped to `0.0..=1.0`
- depth threshold slider, clamped to `0.0..=1.0`

Invalid finite or out-of-range float values coming from UI state should be restored to `RtSettings::default()` values, matching the existing `LightingSettings` sanitation style.

### Sampling Panel

Keep the existing VPT ReSTIR-DI and Area ReSTIR groups.

Add an "RT ReSTIR" group:

- RT ReSTIR-DI toggle
- RT ReSTIR-DI spatial reuse toggle
- RT ReSTIR-DI spatial sample slider, clamped to `0..=8`
- RT ReSTIR-GI toggle

RT GI currently has only the `restir_gi_enabled` setting in `RtSettings`; do not introduce a second UI-owned `RestirGiSettings` path in this phase.

### Debug Panel

Add an RT debug combo bound to `RtSettings.debug_view`:

- `Off`
- `Surface`
- `Hit Distance`
- `History Valid`
- `Direct Reservoir`
- `Indirect Reservoir`
- `Temporal`

RT debug selection should not mutate VPT debug, VPT ReSTIR-DI debug, or Area ReSTIR debug state. These are separate renderer debug surfaces.

## Data Flow

1. `RevolumetricApp::build_egui_frame` passes `&mut self.rt_settings` into `EditorUiFrameState`.
2. `EditorUi::show` forwards the same mutable settings into top bar, inspector, console, and overlay helpers as needed.
3. UI controls mutate `RtSettings` directly.
4. `RevolumetricApp::runtime_settings` copies the updated `RtSettings` into `RuntimeSettings`.
5. `RenderRuntime::ensure_passes` and `RtRuntimePipeline::record_and_execute_frame` consume the updated settings on the next frame.

No new global state, command queue, or event bus is introduced.

## Testing Strategy

Tests should be written before implementation.

Required focused tests:

- editor helper tests for clamping RT history length and spatial sample count
- editor helper tests for sanitizing invalid RT temporal thresholds
- editor helper tests for RT debug labels and option coverage
- source contract test proving `EditorUiFrameState` carries mutable `RtSettings`
- source contract test proving `build_egui_frame` passes `&mut self.rt_settings`
- source contract tests proving Render, Sampling, Debug, top bar, and console expose the expected RT controls

Required verification:

- focused tests for the new editor/app contracts
- `cargo fmt --check`
- `REVOLUMETRIC_SHADER_COMPILE=skip cargo test --lib`
- `git diff --check`

Strict shader compilation is desirable if local `slangc` is available, but this phase does not require new shader code.

## Risks

- The UI will show requested render mode, not actual fallback backend. Users can still request RT on unsupported hardware and runtime will fall back to VPT with a log warning. A later status readback phase should make that visible in UI.
- `src/editor/ui.rs` is already a large file. This phase keeps changes local but may make the file more crowded. A later editor-module split is reasonable after RT controls are covered by tests.
- RT GI has a separate `RestirGiSettings` module, but active runtime control is currently in `RtSettings`. This phase intentionally follows the active runtime setting path to avoid a second, disconnected UI state.

## Out Of Scope

- Changing RT pass ordering.
- Changing RT shader code.
- Changing RT AS build/update behavior.
- Adding actual backend/capability readback to the editor.
- Creating a new standalone RT editor panel.
- Introducing a separate mutable `RestirGiSettings` path into app/runtime UI state.
