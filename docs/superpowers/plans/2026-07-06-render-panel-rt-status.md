# Render Panel RT Status Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Surface actual RT backend status inside the Render inspector panel.

**Architecture:** Reuse `RenderRuntimeStatus` and existing label helpers. Add one small notice helper so source-contract tests and behavior tests can cover fallback messaging without constructing egui.

**Tech Stack:** Rust, egui, source-contract tests, helper tests.

---

### Task 1: Render Panel Runtime Status

**Files:**
- Modify: `src/editor/ui.rs`

- [ ] **Step 1: Write failing tests**

Add tests in `src/editor/ui.rs`:

```rust
#[test]
fn render_panel_exposes_runtime_backend_status_and_rt_fallback_notice() {
    let source = crate::render::source_checks::read_source("src/editor/ui.rs");
    let render_panel = source
        .split("fn show_render_panel")
        .nth(1)
        .expect("render panel should exist")
        .split("fn show_restir_panel")
        .next()
        .expect("render panel should end before sampling panel");

    for token in [
        "runtime_status: Option<RenderRuntimeStatus>",
        "runtime_status_backend_label(runtime_status)",
        "rt_supported_label(runtime_status)",
        "rt_backend_notice(lighting.render_mode, runtime_status)",
    ] {
        assert!(render_panel.contains(token), "render panel missing {token}");
    }
}

#[test]
fn rt_backend_notice_reports_fallback_and_unsupported_states() {
    let rt_active = RenderRuntimeStatus {
        actual_backend: RenderBackend::Rt,
        rt_supported: true,
    };
    let rt_fallback = RenderRuntimeStatus {
        actual_backend: RenderBackend::Vpt,
        rt_supported: true,
    };
    let rt_unsupported = RenderRuntimeStatus {
        actual_backend: RenderBackend::Vpt,
        rt_supported: false,
    };

    assert_eq!(rt_backend_notice(RenderMode::Rt, Some(rt_active)), None);
    assert_eq!(
        rt_backend_notice(RenderMode::Rt, Some(rt_fallback)),
        Some("RT requested; VPT backend is active")
    );
    assert_eq!(
        rt_backend_notice(RenderMode::Auto, Some(rt_unsupported)),
        Some("RT unsupported on this device")
    );
    assert_eq!(rt_backend_notice(RenderMode::Vpt, None), None);
}
```

- [ ] **Step 2: Verify tests fail**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'
cargo test editor::ui::tests::render_panel_exposes_runtime_backend_status_and_rt_fallback_notice --lib
```

Expected: fail because `show_render_panel` does not accept runtime status yet.

- [ ] **Step 3: Thread runtime status into render panel**

Change `show_inspector` to accept `runtime_status: Option<RenderRuntimeStatus>` and pass it to `show_render_panel`.

Change the call site in `EditorUi::show`:

```rust
self.show_inspector(ctx, lighting, runtime_status, rt, restir_di, area_restir);
```

- [ ] **Step 4: Add render panel status UI and helper**

In `show_render_panel`, add labels for actual backend and RT support near the renderer combo. Add:

```rust
fn rt_backend_notice(
    requested: RenderMode,
    status: Option<RenderRuntimeStatus>,
) -> Option<&'static str> {
    let status = status?;
    if !status.rt_supported {
        return Some("RT unsupported on this device");
    }
    if requested == RenderMode::Rt && status.actual_backend != RenderBackend::Rt {
        return Some("RT requested; VPT backend is active");
    }
    None
}
```

- [ ] **Step 5: Verify focused editor tests**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'
cargo test editor::ui::tests::render_panel_exposes_runtime_backend_status_and_rt_fallback_notice editor::ui::tests::rt_backend_notice_reports_fallback_and_unsupported_states --lib
```

Expected: both tests pass.

- [ ] **Step 6: Commit**

```powershell
git add src/editor/ui.rs docs/superpowers/specs/2026-07-06-render-panel-rt-status-design.md docs/superpowers/plans/2026-07-06-render-panel-rt-status.md
git commit -m "feat: show RT fallback status in render panel"
```

---

## Self-Review

- Spec coverage: The plan covers runtime status threading, visible Render panel status, fallback helper, and tests.
- Placeholder scan: No TODO/TBD/deferred implementation placeholders remain.
- Type consistency: `RenderRuntimeStatus`, `RenderBackend`, `RenderMode`, and `rt_backend_notice` names match existing code and planned tests.
