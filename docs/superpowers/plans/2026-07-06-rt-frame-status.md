# RT Frame Status Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Surface RT pass readiness from the RT runtime pipeline into the editor UI.

**Architecture:** `RtRuntimePipeline` exposes an immutable `RtFrameStatus` snapshot of its existing frame state. `RenderRuntimeStatus` carries that optional snapshot, and `src/editor/ui.rs` formats it into compact labels without owning or inferring backend state.

**Tech Stack:** Rust 2024, egui 0.30, existing source-contract tests, `cargo test --lib`, `cargo fmt`, `cargo clippy`.

---

## File Structure

- Modify `src/render/rt_pipeline.rs`: add `RtFrameStatus`, `RtRuntimePipeline::frame_status()`, and tests that verify field mapping.
- Modify `src/render/runtime.rs`: add `rt_frame_status: Option<RtFrameStatus>` to `RenderRuntimeStatus`, wire `RenderRuntime::status()`, and update runtime source-contract tests.
- Modify `src/editor/ui.rs`: add RT frame status labels in top bar, render panel, and console, plus helper/source tests.

## Task 1: RT Pipeline Snapshot

- [ ] **Step 1: Write failing tests in `src/render/rt_pipeline.rs`**

Add tests that instantiate `RtRuntimePipeline`, mutate `frame_state`, and expect `frame_status()` to return a public snapshot:

```rust
#[test]
fn rt_pipeline_frame_status_defaults_to_not_ready() {
    let pipeline = RtRuntimePipeline::new();

    assert_eq!(pipeline.frame_status(), RtFrameStatus::default());
}

#[test]
fn rt_pipeline_frame_status_snapshots_internal_frame_state() {
    let mut pipeline = RtRuntimePipeline::new();
    pipeline.frame_state.surface_initialized = true;
    pipeline.frame_state.restir_di_history_initialized = true;
    pipeline.frame_state.restir_gi_history_initialized = true;
    pipeline.frame_state.direct_lighting_initialized = true;
    pipeline.frame_state.temporal_initialized = true;
    pipeline.frame_state.resolve_initialized = true;

    assert_eq!(
        pipeline.frame_status(),
        RtFrameStatus {
            frame_resources_ready: false,
            surface_ready: true,
            restir_di_history_ready: true,
            restir_gi_history_ready: true,
            direct_lighting_ready: true,
            temporal_ready: true,
            resolve_ready: true,
        }
    );
}
```

- [ ] **Step 2: Run RED**

Run:

```powershell
REVOLUMETRIC_SHADER_COMPILE=skip cargo test render::rt_pipeline::tests::rt_pipeline_frame_status_defaults_to_not_ready render::rt_pipeline::tests::rt_pipeline_frame_status_snapshots_internal_frame_state --lib
```

Expected: failure because `RtFrameStatus` and `RtRuntimePipeline::frame_status()` do not exist yet.

- [ ] **Step 3: Implement snapshot**

Add `RtFrameStatus` near `RtPipelineFrameState`:

```rust
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RtFrameStatus {
    pub frame_resources_ready: bool,
    pub surface_ready: bool,
    pub restir_di_history_ready: bool,
    pub restir_gi_history_ready: bool,
    pub direct_lighting_ready: bool,
    pub temporal_ready: bool,
    pub resolve_ready: bool,
}
```

Add `RtRuntimePipeline::frame_status()`:

```rust
pub fn frame_status(&self) -> RtFrameStatus {
    RtFrameStatus {
        frame_resources_ready: self.has_frame_resources(),
        surface_ready: self.frame_state.surface_initialized,
        restir_di_history_ready: self.frame_state.restir_di_history_initialized,
        restir_gi_history_ready: self.frame_state.restir_gi_history_initialized,
        direct_lighting_ready: self.frame_state.direct_lighting_initialized,
        temporal_ready: self.frame_state.temporal_initialized,
        resolve_ready: self.frame_state.resolve_initialized,
    }
}
```

- [ ] **Step 4: Run GREEN**

Run the same focused command and expect both tests to pass.

## Task 2: Runtime Status Wiring

- [ ] **Step 1: Write failing runtime tests in `src/render/runtime.rs`**

Update existing `RenderRuntimeStatus` test fixtures with `rt_frame_status: None`, and extend `render_runtime_status_exposes_backend_and_rt_support_for_editor` to require:

```rust
"pub rt_frame_status: Option<RtFrameStatus>"
"rt_frame_status: (self.render_backend == RenderBackend::Rt || self.rt_pipeline.has_frame_resources()).then(|| self.rt_pipeline.frame_status())"
```

- [ ] **Step 2: Run RED**

Run:

```powershell
REVOLUMETRIC_SHADER_COMPILE=skip cargo test render::runtime::tests::render_runtime_status_exposes_backend_and_rt_support_for_editor --lib
```

Expected: failure because `RenderRuntimeStatus` does not expose RT frame status yet.

- [ ] **Step 3: Implement runtime status wiring**

Import `RtFrameStatus` from `src/render/rt_pipeline.rs`, add `rt_frame_status: Option<RtFrameStatus>` to `RenderRuntimeStatus`, and set it in `RenderRuntime::status()`:

```rust
rt_frame_status: (self.render_backend == RenderBackend::Rt || self.rt_pipeline.has_frame_resources())
    .then(|| self.rt_pipeline.frame_status()),
```

- [ ] **Step 4: Run GREEN**

Run the same focused runtime test and expect it to pass.

## Task 3: Editor UI Labels

- [ ] **Step 1: Write failing editor tests in `src/editor/ui.rs`**

Add helper tests that build `RenderRuntimeStatus` values with `rt_frame_status: None` and `Some(RtFrameStatus { ... })`, then verify:

```rust
assert_eq!(rt_frame_status_label(None), "pending");
assert_eq!(rt_frame_status_label(Some(inactive)), "inactive");
assert_eq!(rt_frame_status_label(Some(warming)), "warming");
assert_eq!(rt_frame_status_label(Some(ready)), "ready");
```

Add source-contract checks for these UI tokens:

```rust
"rt_frame_status_label(runtime_status)"
"rt_frame_surface_label(runtime_status)"
"rt_frame_direct_lighting_label(runtime_status)"
"rt_frame_temporal_label(runtime_status)"
"rt_frame_resolve_label(runtime_status)"
"rt_frame_restir_di_history_label(runtime_status)"
"rt_frame_restir_gi_history_label(runtime_status)"
"rt_frame={}"
"rt_surface={}"
"rt_direct={}"
"rt_temporal_ready={}"
"rt_resolve={}"
"rt_di_history={}"
"rt_gi_history={}"
```

- [ ] **Step 2: Run RED**

Run:

```powershell
REVOLUMETRIC_SHADER_COMPILE=skip cargo test editor::ui::tests::runtime_status_labels_cover_backend_pending_and_rt_frame_states editor::ui::tests::top_bar_render_panel_and_console_report_rt_frame_status --lib
```

Expected: failure because the UI helpers and display tokens are not implemented yet.

- [ ] **Step 3: Implement editor helpers and UI tokens**

Import `RtFrameStatus`, add label helpers, add `rt_frame` to the top bar, add a compact readiness row in `show_render_panel`, and add machine-readable console tokens.

- [ ] **Step 4: Run GREEN**

Run the same focused editor command and expect both tests to pass.

## Task 4: Full Verification And Commit

- [ ] Run `cargo fmt --check`; if it fails due formatting, run `cargo fmt` and rerun `cargo fmt --check`.
- [ ] Run `REVOLUMETRIC_SHADER_COMPILE=skip cargo test --lib`.
- [ ] Run `REVOLUMETRIC_SHADER_COMPILE=skip cargo clippy --all-targets -- -D warnings`.
- [ ] Run `REVOLUMETRIC_SHADER_COMPILE=strict cargo test --lib`.
- [ ] Run `git diff --check`.
- [ ] Run `git status --short --branch`.
- [ ] Commit the focused change with message `feat: surface RT frame status in editor`.
