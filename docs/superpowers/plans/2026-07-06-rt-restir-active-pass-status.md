# RT ReSTIR Active Pass Status Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Surface last-frame RT ReSTIR-DI and RT ReSTIR-GI pass activity through `RtFrameStatus` and the editor UI.

**Architecture:** `RtRuntimePipeline` already computes `rt_restir_di_rendered` and `rt_restir_gi_rendered` while recording the RT graph. Persist those booleans in `RtPipelineFrameState`, snapshot them through `RtFrameStatus`, and format them in `src/editor/ui.rs` using the existing RT frame boolean-label helper.

**Tech Stack:** Rust 2024, egui 0.30, existing source-contract tests, Cargo fmt/test/clippy, PowerShell visual baseline validation.

---

## File Structure

- Modify `src/render/rt_pipeline.rs`: add active optional-pass fields to `RtFrameStatus` and `RtPipelineFrameState`, clear them on history reset/fallback, snapshot them in `frame_status()`, and add source/unit tests.
- Modify `src/editor/ui.rs`: add `rt_frame_restir_di_pass_label` and `rt_frame_restir_gi_pass_label`, display them in the render panel and console, and add helper/source tests.
- No changes to shaders, render graph registration conditions, capture metadata, or visual baseline manifest are part of this slice.

## Task 1: RT Pipeline Active-Pass State

**Files:**
- Modify: `src/render/rt_pipeline.rs`

- [ ] **Step 1: Write failing frame status tests**

In `rt_pipeline_frame_status_snapshots_internal_frame_state`, set:

```rust
pipeline.frame_state.restir_di_rendered = true;
pipeline.frame_state.restir_gi_rendered = true;
```

and extend the expected `RtFrameStatus`:

```rust
restir_di_rendered: true,
restir_gi_rendered: true,
```

Add a reset test next to `rt_pipeline_history_reset_clears_skip_reason`:

```rust
#[test]
fn rt_pipeline_history_reset_clears_active_restir_pass_state() {
    let mut pipeline = RtRuntimePipeline::new();
    pipeline.frame_state.restir_di_rendered = true;
    pipeline.frame_state.restir_gi_rendered = true;

    pipeline.reset_history(7);

    let status = pipeline.frame_status();
    assert!(!status.restir_di_rendered);
    assert!(!status.restir_gi_rendered);
}
```

- [ ] **Step 2: Write failing source-contract test**

Add this test after `rt_pipeline_records_structured_skip_reasons_for_fallbacks`:

```rust
#[test]
fn rt_pipeline_records_active_restir_pass_state() {
    let source = crate::render::source_checks::read_source("src/render/rt_pipeline.rs");
    let record = source
        .split("pub fn record_and_execute_frame")
        .nth(1)
        .expect("RtRuntimePipeline::record_and_execute_frame should exist")
        .split("pub fn destroy")
        .next()
        .expect("record_and_execute_frame should end before destroy");
    let compact = crate::render::source_checks::compact(record);

    for token in [
        "letrt_restir_di_rendered=false;",
        "letrt_restir_gi_rendered=false;",
        "rt_restir_di_rendered=rt_restir_di_reservoir_resource.is_some();",
        "rt_restir_gi_rendered=rt_restir_gi_reservoir_resource.is_some();",
        "self.frame_state.restir_di_rendered=rt_graph_rendered&&rt_restir_di_rendered;",
        "self.frame_state.restir_gi_rendered=rt_graph_rendered&&rt_restir_gi_rendered;",
    ] {
        assert!(
            compact.contains(token),
            "RT pipeline missing active RT ReSTIR pass state token {token}"
        );
    }
}
```

- [ ] **Step 3: Run RED**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib rt_pipeline
```

Expected result: failure because `restir_di_rendered` and `restir_gi_rendered` are not fields on `RtPipelineFrameState` or `RtFrameStatus`.

- [ ] **Step 4: Implement active-pass fields**

Add fields to `RtFrameStatus`:

```rust
pub restir_di_rendered: bool,
pub restir_gi_rendered: bool,
```

Add fields to `RtPipelineFrameState`:

```rust
pub restir_di_rendered: bool,
pub restir_gi_rendered: bool,
```

Clear both in `RtPipelineFrameState::reset_history`:

```rust
self.restir_di_rendered = false;
self.restir_gi_rendered = false;
```

Snapshot both in `frame_status()`:

```rust
restir_di_rendered: self.frame_state.restir_di_rendered,
restir_gi_rendered: self.frame_state.restir_gi_rendered,
```

Persist them after graph execution:

```rust
self.frame_state.restir_di_rendered = rt_graph_rendered && rt_restir_di_rendered;
self.frame_state.restir_gi_rendered = rt_graph_rendered && rt_restir_gi_rendered;
```

- [ ] **Step 5: Run GREEN**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib rt_pipeline
```

Expected result: RT pipeline tests pass.

## Task 2: Editor Active-Pass Labels

**Files:**
- Modify: `src/editor/ui.rs`

- [ ] **Step 1: Write failing label helper test**

Add this test after `rt_frame_skip_reason_labels_cover_pending_inactive_none_and_reasons`:

```rust
#[test]
fn rt_frame_restir_active_pass_labels_cover_true_false_and_unknown() {
    let inactive = RenderRuntimeStatus {
        actual_backend: RenderBackend::Vpt,
        rt_supported: true,
        rt_frame_status: None,
    };
    let active = RenderRuntimeStatus {
        actual_backend: RenderBackend::Rt,
        rt_supported: true,
        rt_frame_status: Some(RtFrameStatus {
            restir_di_rendered: true,
            restir_gi_rendered: false,
            ..RtFrameStatus::default()
        }),
    };

    assert_eq!(rt_frame_restir_di_pass_label(Some(active)), "true");
    assert_eq!(rt_frame_restir_gi_pass_label(Some(active)), "false");
    assert_eq!(rt_frame_restir_di_pass_label(Some(inactive)), "unknown");
    assert_eq!(rt_frame_restir_gi_pass_label(None), "unknown");
}
```

- [ ] **Step 2: Extend failing source tests**

In `top_bar_render_panel_and_console_report_rt_frame_status`, add render panel tokens:

```rust
"rt_frame_restir_di_pass_label(runtime_status)",
"rt_frame_restir_gi_pass_label(runtime_status)",
```

and console tokens:

```rust
"rt_restir_di_pass={}",
"rt_restir_gi_pass={}",
```

- [ ] **Step 3: Run RED**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib rt_frame
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib top_bar_render_panel_and_console_report_rt_frame_status
```

Expected result: failure because the two helper functions and UI display tokens do not exist.

- [ ] **Step 4: Implement UI helpers and display tokens**

Add helpers near the existing RT frame label helpers:

```rust
fn rt_frame_restir_di_pass_label(status: Option<RenderRuntimeStatus>) -> &'static str {
    rt_frame_bool_label(status, |frame_status| frame_status.restir_di_rendered)
}

fn rt_frame_restir_gi_pass_label(status: Option<RenderRuntimeStatus>) -> &'static str {
    rt_frame_bool_label(status, |frame_status| frame_status.restir_gi_rendered)
}
```

Add render panel labels in the existing `RT ready` row:

```rust
ui.monospace(format!(
    "di_pass {}",
    rt_frame_restir_di_pass_label(runtime_status)
));
ui.monospace(format!(
    "gi_pass {}",
    rt_frame_restir_gi_pass_label(runtime_status)
));
```

Add console tokens after `rt_gi_history`:

```rust
ui.monospace(format!(
    "rt_restir_di_pass={}",
    rt_frame_restir_di_pass_label(runtime_status)
));
ui.monospace(format!(
    "rt_restir_gi_pass={}",
    rt_frame_restir_gi_pass_label(runtime_status)
));
```

- [ ] **Step 5: Run GREEN**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib rt_frame
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib top_bar_render_panel_and_console_report_rt_frame_status
```

Expected result: editor-focused tests pass.

## Task 3: Verification, Review, And Commit

**Files:**
- Review: `src/render/rt_pipeline.rs`, `src/editor/ui.rs`

- [ ] **Step 1: Run focused verification**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib rt_pipeline
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib rt_frame
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib top_bar_render_panel_and_console_report_rt_frame_status
```

Expected result: focused RT pipeline and editor tests pass.

- [ ] **Step 2: Run full verification**

Run:

```powershell
cargo fmt --check
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo clippy --all-targets -- -D warnings
$env:REVOLUMETRIC_SHADER_COMPILE='strict'; cargo test --lib
.\run\validate-visual-baseline.ps1 -Rt
git diff --check
git status --short --branch
```

Expected result: each command exits with code 0. Windows CRLF warnings from `git diff --check` are acceptable if the exit code is 0.

- [ ] **Step 3: Request read-only code review**

Ask a reviewer to inspect this slice only:

```text
Review the RT ReSTIR active-pass status changes. Check that active pass booleans snapshot last-frame RT ReSTIR-DI/GI participation, clear on fallback/reset, are distinct from history-ready booleans, and are displayed only in render panel/console. Do not edit files.
```

Fix Critical or Important feedback and rerun relevant verification.

- [ ] **Step 4: Commit**

Run:

```powershell
git add src/render/rt_pipeline.rs src/editor/ui.rs
git commit -m "feat: surface RT ReSTIR active pass status"
```
