# RT Runtime Status Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Display actual runtime backend status in the editor and make `RenderRuntime` respond to live renderer mode changes.

**Architecture:** Keep backend authority inside `RenderRuntime`. Add a tiny status snapshot, refresh the selected backend from current runtime settings before pass selection, and pass the status snapshot through `RevolumetricApp` into the existing editor frame state.

**Tech Stack:** Rust 2024, egui 0.30, existing source-contract tests, `cargo test --lib`, `cargo fmt`, `cargo clippy`.

---

## File Structure

- Modify `src/render/runtime.rs`: add `RenderRuntimeStatus`, `RenderRuntime::status()`, backend refresh helper, and runtime source tests.
- Modify `src/app.rs`: pass `self.render_runtime.as_ref().map(RenderRuntime::status)` into `EditorUiFrameState`, and add an app source test.
- Modify `src/editor/ui.rs`: accept optional runtime status, show actual backend and RT support in top bar and console, and add label/source tests.

## Task 1: Runtime Backend Refresh And Status Tests

- [ ] Add failing runtime tests in `src/render/runtime.rs` for `RenderRuntimeStatus`, `RenderRuntime::status()`, `refresh_render_backend`, `render_frame`, and `resize_pipeline_to_swapchain`.
- [ ] Run focused runtime tests with `REVOLUMETRIC_SHADER_COMPILE=skip cargo test render::runtime::tests::render_runtime_status_exposes_backend_and_rt_support_for_editor render::runtime::tests::render_frame_refreshes_backend_from_current_settings_before_pass_selection render::runtime::tests::resize_refreshes_backend_from_current_settings --lib` and confirm failure.
- [ ] Implement `RenderRuntimeStatus`, `status()`, and `refresh_render_backend()`.
- [ ] Call `refresh_render_backend(input.settings.lighting.render_mode)` in `render_frame` after `begin_frame` and before backend-change reset/pass selection.
- [ ] Call `refresh_render_backend(settings.lighting.render_mode)` at the start of `resize_pipeline_to_swapchain`.
- [ ] Re-run the focused runtime tests and confirm pass.

## Task 2: App And Editor Wiring Tests

- [ ] Add failing app source test proving `build_egui_frame` passes `runtime_status: self.render_runtime.as_ref().map(RenderRuntime::status)`.
- [ ] Add failing editor tests proving `EditorUiFrameState` carries `Option<RenderRuntimeStatus>`, top bar/console show `actual_backend`, and console shows `rt_supported`.
- [ ] Add helper test for `render_backend_label` and `runtime_status_backend_label`.
- [ ] Run focused app/editor tests with `REVOLUMETRIC_SHADER_COMPILE=skip cargo test app::tests::app_passes_runtime_status_to_editor_ui_frame_state editor::ui::tests::editor_frame_state_carries_runtime_status editor::ui::tests::top_bar_and_console_report_runtime_backend_status editor::ui::tests::runtime_status_labels_cover_backend_and_pending_states --lib` and confirm failure.
- [ ] Implement app/editor wiring and labels.
- [ ] Re-run the focused app/editor tests and confirm pass.

## Task 3: Final Verification And Commit

- [ ] Run `cargo fmt --check`; if it fails, run `cargo fmt` and rerun the check.
- [ ] Run `REVOLUMETRIC_SHADER_COMPILE=skip cargo test --lib`.
- [ ] Run `REVOLUMETRIC_SHADER_COMPILE=skip cargo clippy --all-targets -- -D warnings`.
- [ ] Run `REVOLUMETRIC_SHADER_COMPILE=strict cargo test --lib`.
- [ ] Run `git diff --check` and `git status --short`.
- [ ] Commit docs and implementation with focused messages.
