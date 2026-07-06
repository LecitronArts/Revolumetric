# RT Frame Skip Reason Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Surface the last RT frame's structured skip reason in the editor UI.

**Architecture:** `RtRuntimePipeline` records a root-cause `RtFrameSkipReason` in its frame state whenever it falls back before RT resolve. `RtFrameStatus` snapshots that optional reason through the existing `RenderRuntimeStatus.rt_frame_status` path, and `src/editor/ui.rs` formats it into stable compact labels.

**Tech Stack:** Rust 2024, egui 0.30, existing source-contract tests, PowerShell validation commands, Cargo fmt/test/clippy.

---

## File Structure

- Modify `src/render/rt_pipeline.rs`: add `RtFrameSkipReason`, add `skip_reason` to `RtFrameStatus` and `RtPipelineFrameState`, set it in fallback branches, clear it on successful RT resolve registration, and add tests.
- Modify `src/editor/ui.rs`: import `RtFrameSkipReason`, add `rt_frame_skip_reason_label`, display it in top bar, render panel, and console, and add helper/source tests.
- No changes are expected in `src/render/runtime.rs` because `RenderRuntimeStatus` already carries `Option<RtFrameStatus>`.

## Task 1: RT Pipeline Structured Skip Reason

- [ ] **Step 1: Write failing tests in `src/render/rt_pipeline.rs`**

Add `RtFrameSkipReason` expectations beside the existing RT frame status tests:

```rust
#[test]
fn rt_pipeline_frame_status_snapshots_skip_reason() {
    let mut pipeline = RtRuntimePipeline::new();
    pipeline.frame_state.skip_reason = Some(RtFrameSkipReason::UcvhUploadPending);

    assert_eq!(
        pipeline.frame_status().skip_reason,
        Some(RtFrameSkipReason::UcvhUploadPending)
    );
}

#[test]
fn rt_pipeline_records_structured_skip_reasons_for_fallbacks() {
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
        "skip_reason.get_or_insert(RtFrameSkipReason::UcvhUploadPending)",
        "skip_reason.get_or_insert(RtFrameSkipReason::CpuUcvhSceneMissing)",
        "skip_reason.get_or_insert(RtFrameSkipReason::AccelerationStructureLoaderMissing)",
        "skip_reason.get_or_insert(RtFrameSkipReason::AccelerationStructureRebuildFailed)",
        "skip_reason.get_or_insert(RtFrameSkipReason::RequiredPassesMissing)",
        "skip_reason.get_or_insert(RtFrameSkipReason::AccelerationStructureMissing)",
        "skip_reason.get_or_insert(RtFrameSkipReason::UcvhGpuDescriptorsMissing)",
        "self.frame_state.skip_reason=ifrt_graph_rendered{None}else{skip_reason}",
    ] {
        assert!(
            compact.contains(token),
            "RT pipeline missing structured skip reason token {token}"
        );
    }
}
```

Update `rt_pipeline_frame_status_snapshots_internal_frame_state` expected `RtFrameStatus` with:

```rust
skip_reason: None,
```

- [ ] **Step 2: Run RED**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib rt_pipeline_frame_status_snapshots_skip_reason rt_pipeline_records_structured_skip_reasons_for_fallbacks rt_pipeline_frame_status_snapshots_internal_frame_state
```

Expected: compile or assertion failure because `RtFrameSkipReason` and `skip_reason` do not exist yet.

- [ ] **Step 3: Implement RT skip reason state**

Add the enum near `RtFrameStatus`:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

Add `skip_reason: Option<RtFrameSkipReason>` to `RtFrameStatus` and `RtPipelineFrameState`, then copy it in `frame_status()`:

```rust
skip_reason: self.frame_state.skip_reason,
```

In `record_and_execute_frame`, create a local before AS rebuild:

```rust
let mut skip_reason = None;
```

Set root-cause reasons with `skip_reason.get_or_insert(...)`:

```rust
skip_reason.get_or_insert(RtFrameSkipReason::UcvhUploadPending);
skip_reason.get_or_insert(RtFrameSkipReason::CpuUcvhSceneMissing);
skip_reason.get_or_insert(RtFrameSkipReason::AccelerationStructureLoaderMissing);
skip_reason.get_or_insert(RtFrameSkipReason::AccelerationStructureRebuildFailed);
skip_reason.get_or_insert(RtFrameSkipReason::RequiredPassesMissing);
skip_reason.get_or_insert(RtFrameSkipReason::AccelerationStructureMissing);
skip_reason.get_or_insert(RtFrameSkipReason::UcvhGpuDescriptorsMissing);
```

After graph execution updates readiness state, persist:

```rust
self.frame_state.skip_reason = if rt_graph_rendered { None } else { skip_reason };
```

- [ ] **Step 4: Run GREEN**

Run the same focused command and expect the tests to pass.

## Task 2: Editor Skip Reason Labels

- [ ] **Step 1: Write failing editor tests in `src/editor/ui.rs`**

Import `RtFrameSkipReason` in the test module and add helper coverage:

```rust
#[test]
fn rt_frame_skip_reason_labels_cover_pending_inactive_none_and_reasons() {
    let inactive = RenderRuntimeStatus {
        actual_backend: RenderBackend::Vpt,
        rt_supported: true,
        rt_frame_status: None,
    };
    let ready = RenderRuntimeStatus {
        actual_backend: RenderBackend::Rt,
        rt_supported: true,
        rt_frame_status: Some(RtFrameStatus::default()),
    };

    assert_eq!(rt_frame_skip_reason_label(None), "pending");
    assert_eq!(rt_frame_skip_reason_label(Some(inactive)), "inactive");
    assert_eq!(rt_frame_skip_reason_label(Some(ready)), "none");

    for (reason, label) in [
        (RtFrameSkipReason::UcvhUploadPending, "ucvh_upload_pending"),
        (RtFrameSkipReason::CpuUcvhSceneMissing, "cpu_ucvh_missing"),
        (
            RtFrameSkipReason::AccelerationStructureLoaderMissing,
            "as_loader_missing",
        ),
        (
            RtFrameSkipReason::AccelerationStructureRebuildFailed,
            "as_rebuild_failed",
        ),
        (RtFrameSkipReason::AccelerationStructureMissing, "as_missing"),
        (RtFrameSkipReason::UcvhGpuDescriptorsMissing, "ucvh_gpu_missing"),
        (
            RtFrameSkipReason::RequiredPassesMissing,
            "required_passes_missing",
        ),
    ] {
        let status = RenderRuntimeStatus {
            actual_backend: RenderBackend::Rt,
            rt_supported: true,
            rt_frame_status: Some(RtFrameStatus {
                skip_reason: Some(reason),
                ..RtFrameStatus::default()
            }),
        };
        assert_eq!(rt_frame_skip_reason_label(Some(status)), label);
    }
}
```

Extend `top_bar_render_panel_and_console_report_rt_frame_status` source checks with:

```rust
"rt_frame_skip_reason_label(runtime_status)"
"RT reason"
"rt_skip_reason={}"
```

- [ ] **Step 2: Run RED**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib rt_frame_skip_reason_labels_cover_pending_inactive_none_and_reasons top_bar_render_panel_and_console_report_rt_frame_status
```

Expected: failure because the UI helper and display tokens do not exist.

- [ ] **Step 3: Implement UI labels and display tokens**

Update the import:

```rust
use crate::render::rt_pipeline::{RtFrameSkipReason, RtFrameStatus};
```

Add the helper near existing RT frame label helpers:

```rust
fn rt_frame_skip_reason_label(status: Option<RenderRuntimeStatus>) -> &'static str {
    let Some(status) = status else {
        return "pending";
    };
    let Some(frame_status) = status.rt_frame_status else {
        return "inactive";
    };
    match frame_status.skip_reason {
        None => "none",
        Some(RtFrameSkipReason::UcvhUploadPending) => "ucvh_upload_pending",
        Some(RtFrameSkipReason::CpuUcvhSceneMissing) => "cpu_ucvh_missing",
        Some(RtFrameSkipReason::AccelerationStructureLoaderMissing) => "as_loader_missing",
        Some(RtFrameSkipReason::AccelerationStructureRebuildFailed) => "as_rebuild_failed",
        Some(RtFrameSkipReason::AccelerationStructureMissing) => "as_missing",
        Some(RtFrameSkipReason::UcvhGpuDescriptorsMissing) => "ucvh_gpu_missing",
        Some(RtFrameSkipReason::RequiredPassesMissing) => "required_passes_missing",
    }
}
```

Display it in the existing UI paths:

```rust
ui.label(format!("rt_reason {}", rt_frame_skip_reason_label(runtime_status)));
```

```rust
ui.label("RT reason");
ui.monospace(rt_frame_skip_reason_label(runtime_status));
```

```rust
ui.monospace(format!(
    "rt_skip_reason={}",
    rt_frame_skip_reason_label(runtime_status)
));
```

- [ ] **Step 4: Run GREEN**

Run the same focused editor command and expect the tests to pass.

## Task 3: Verification And Commit

- [ ] Run `cargo fmt --check`; if it fails due formatting, run `cargo fmt` and rerun `cargo fmt --check`.
- [ ] Run `$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib`.
- [ ] Run `$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo clippy --all-targets -- -D warnings`.
- [ ] Run `$env:REVOLUMETRIC_SHADER_COMPILE='strict'; cargo test --lib`.
- [ ] Run `.\run\validate-visual-baseline.ps1 -Rt`.
- [ ] Run `git diff --check`.
- [ ] Run `git status --short --branch`.
- [ ] Commit with message `feat: surface RT frame skip reason`.
