# Render Panel RT Status Design

## Goal

Show the actual RT backend status in the Render inspector panel where users choose the requested renderer mode.

## Current Facts

- The top bar and console already display requested mode, actual backend, and RT support.
- The Render inspector panel owns the renderer mode combo and RT temporal controls, but it does not receive `RenderRuntimeStatus`.
- When RT is requested on unsupported hardware, the actual backend can be VPT while RT settings remain visible and mutable.

## Design

Thread `Option<RenderRuntimeStatus>` from `EditorUi::show` into `show_inspector`, then into `show_render_panel`.

In `show_render_panel`, display:

- actual backend label using the existing runtime status label helper
- RT support label using the existing support helper
- a concise warning label when `RenderMode::Rt` is requested but the actual backend is `RenderBackend::Vpt`
- a concise warning label when runtime reports `rt_supported == false`

Do not disable RT controls. Users should still be able to preset RT settings before switching devices or before the runtime catches up on the next frame.

## Non-Goals

- Do not change backend resolution logic.
- Do not add a new panel.
- Do not add GPU runtime tests.
- Do not change RT setting defaults.

## Testing Strategy

Add source-contract and helper tests in `src/editor/ui.rs`:

- Render panel accepts runtime status.
- Render panel displays actual backend and RT support status.
- Render panel uses a fallback notice helper.
- The helper returns a notice for requested RT with actual VPT and unsupported RT, but not for normal RT-active state.

Run focused editor UI tests, then the project verification gates.
