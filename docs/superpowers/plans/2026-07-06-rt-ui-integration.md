# RT UI Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose existing hardware RT runtime settings in the egui editor and prove the UI-to-runtime setting flow with focused tests.

**Architecture:** Keep the current direct mutable-settings editor pattern. Pass `RtSettings` from `RevolumetricApp` into `EditorUiFrameState`, then thread it through existing top bar, inspector, console, Render, Sampling, and Debug helpers. Add small editor helper functions for RT labels, option lists, clamping, and sanitation so behavior can be tested without a live Vulkan/egui frame.

**Tech Stack:** Rust 2024, egui 0.30, existing `LightingSettings`, `RtSettings`, `cargo test --lib`, `cargo fmt`, source contract tests.

---

## File Structure

- Modify `src/editor/ui.rs`: imports, frame state, panel signatures, RT controls, RT labels/options, clamping/sanitation helpers, and editor unit/source tests.
- Modify `src/app.rs`: pass `&mut self.rt_settings` to `EditorUiFrameState` and add an app wiring source test.
- No shader, RT pass, render graph, or runtime backend changes are needed for this phase.

## Task 1: Add RT UI Helper Tests

**Files:**
- Modify: `src/editor/ui.rs`

- [ ] **Step 1: Write failing helper tests**

Add these tests inside `src/editor/ui.rs` `#[cfg(test)] mod tests`:

```rust
use crate::render::rt_settings::{RtDebugView, RtSettings};

#[test]
fn editor_rt_history_length_control_clamps_to_runtime_range() {
    assert_eq!(clamp_rt_history_length(0), 1);
    assert_eq!(clamp_rt_history_length(20), 20);
    assert_eq!(clamp_rt_history_length(128), 64);
}

#[test]
fn editor_rt_spatial_sample_control_clamps_to_runtime_range() {
    assert_eq!(clamp_rt_spatial_samples(0), 0);
    assert_eq!(clamp_rt_spatial_samples(4), 4);
    assert_eq!(clamp_rt_spatial_samples(128), 8);
}

#[test]
fn rt_temporal_threshold_sanitizer_restores_invalid_values_to_defaults() {
    let mut settings = RtSettings {
        normal_threshold: f32::NAN,
        depth_threshold: -1.0,
        ..RtSettings::default()
    };

    sanitize_rt_temporal_thresholds(&mut settings);

    assert_eq!(
        settings.normal_threshold,
        RtSettings::default().normal_threshold
    );
    assert_eq!(
        settings.depth_threshold,
        RtSettings::default().depth_threshold
    );
}

#[test]
fn rt_debug_options_cover_every_runtime_debug_view() {
    let views: Vec<RtDebugView> = RT_DEBUG_OPTIONS.iter().map(|(view, _)| *view).collect();

    assert_eq!(
        views,
        vec![
            RtDebugView::Off,
            RtDebugView::Surface,
            RtDebugView::HitDistance,
            RtDebugView::HistoryValid,
            RtDebugView::DirectReservoir,
            RtDebugView::IndirectReservoir,
            RtDebugView::Temporal,
        ]
    );
    assert_eq!(rt_debug_label(RtDebugView::DirectReservoir), "Direct Reservoir");
}
```

- [ ] **Step 2: Run focused tests and confirm they fail**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'
cargo test editor::ui::tests::editor_rt_history_length_control_clamps_to_runtime_range --lib
cargo test editor::ui::tests::editor_rt_spatial_sample_control_clamps_to_runtime_range --lib
cargo test editor::ui::tests::rt_temporal_threshold_sanitizer_restores_invalid_values_to_defaults --lib
cargo test editor::ui::tests::rt_debug_options_cover_every_runtime_debug_view --lib
Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
```

Expected: compile failures because `clamp_rt_history_length`, `clamp_rt_spatial_samples`, `sanitize_rt_temporal_thresholds`, `RT_DEBUG_OPTIONS`, and `rt_debug_label` do not exist yet.

- [ ] **Step 3: Implement helper functions and constants**

Add these helpers in `src/editor/ui.rs` near the existing VPT helper functions:

```rust
pub fn clamp_rt_history_length(value: u32) -> u32 {
    value.clamp(1, 64)
}

pub fn clamp_rt_spatial_samples(value: u32) -> u32 {
    value.clamp(0, 8)
}

pub fn sanitize_rt_temporal_thresholds(settings: &mut RtSettings) {
    let defaults = RtSettings::default();
    if !settings.normal_threshold.is_finite()
        || settings.normal_threshold < 0.0
        || settings.normal_threshold > 1.0
    {
        settings.normal_threshold = defaults.normal_threshold;
    }
    if !settings.depth_threshold.is_finite()
        || settings.depth_threshold < 0.0
        || settings.depth_threshold > 1.0
    {
        settings.depth_threshold = defaults.depth_threshold;
    }
}

fn rt_debug_label(debug_view: RtDebugView) -> &'static str {
    RT_DEBUG_OPTIONS
        .iter()
        .find_map(|(view, label)| (*view == debug_view).then_some(*label))
        .unwrap_or("Unknown")
}

const RT_DEBUG_OPTIONS: &[(RtDebugView, &str)] = &[
    (RtDebugView::Off, "Off"),
    (RtDebugView::Surface, "Surface"),
    (RtDebugView::HitDistance, "Hit Distance"),
    (RtDebugView::HistoryValid, "History Valid"),
    (RtDebugView::DirectReservoir, "Direct Reservoir"),
    (RtDebugView::IndirectReservoir, "Indirect Reservoir"),
    (RtDebugView::Temporal, "Temporal"),
];
```

Import `RtDebugView` and `RtSettings` at the top of `src/editor/ui.rs`.

- [ ] **Step 4: Run focused tests and confirm they pass**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'
cargo test editor::ui::tests::editor_rt_history_length_control_clamps_to_runtime_range --lib
cargo test editor::ui::tests::editor_rt_spatial_sample_control_clamps_to_runtime_range --lib
cargo test editor::ui::tests::rt_temporal_threshold_sanitizer_restores_invalid_values_to_defaults --lib
cargo test editor::ui::tests::rt_debug_options_cover_every_runtime_debug_view --lib
Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
```

Expected: all four focused tests pass.

## Task 2: Wire RtSettings Into Editor Frame State

**Files:**
- Modify: `src/editor/ui.rs`
- Modify: `src/app.rs`

- [ ] **Step 1: Write failing source contract tests**

Add this test to `src/editor/ui.rs`:

```rust
#[test]
fn editor_frame_state_carries_mutable_rt_settings() {
    let source = crate::render::source_checks::read_source("src/editor/ui.rs");
    let frame_state = source
        .split("pub struct EditorUiFrameState")
        .nth(1)
        .expect("EditorUiFrameState should exist")
        .split("impl Default for EditorUi")
        .next()
        .expect("EditorUiFrameState should end before EditorUi impls");

    assert!(frame_state.contains("pub rt: &'a mut RtSettings"));
}
```

Add this test to `src/app.rs` tests:

```rust
#[test]
fn app_passes_rt_settings_to_editor_ui_frame_state() {
    let source = crate::render::source_checks::read_source("src/app.rs");
    let build_egui_frame = source
        .split("fn build_egui_frame")
        .nth(1)
        .expect("build_egui_frame should exist")
        .split("#[cfg(target_os = \"android\")]")
        .next()
        .expect("desktop build_egui_frame should end before android variant");

    assert!(build_egui_frame.contains("rt: &mut self.rt_settings"));
}
```

- [ ] **Step 2: Run focused tests and confirm they fail**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'
cargo test editor::ui::tests::editor_frame_state_carries_mutable_rt_settings --lib
cargo test app::tests::app_passes_rt_settings_to_editor_ui_frame_state --lib
Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
```

Expected: assertion failures because the RT field is not wired yet.

- [ ] **Step 3: Implement editor frame state wiring**

Update `src/editor/ui.rs`:

```rust
use crate::render::rt_settings::{RtDebugView, RtSettings};

pub struct EditorUiFrameState<'a> {
    pub lighting: &'a mut LightingSettings,
    pub rt: &'a mut RtSettings,
    pub restir_di: &'a mut RestirDiSettings,
    pub area_restir: &'a mut AreaRestirSettings,
    pub camera: VptCameraFrame,
    pub viewport_extent: [u32; 2],
    pub rendered_frames: u64,
}
```

Destructure `rt` in `EditorUi::show` and pass it to the helper methods that need it.

Update `src/app.rs`:

```rust
EditorUiFrameState {
    lighting: &mut self.lighting_settings,
    rt: &mut self.rt_settings,
    restir_di: &mut self.restir_di_settings,
    area_restir: &mut self.area_restir_settings,
    camera,
    viewport_extent,
    rendered_frames: self.rendered_frames,
}
```

- [ ] **Step 4: Run focused tests and confirm they pass**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'
cargo test editor::ui::tests::editor_frame_state_carries_mutable_rt_settings --lib
cargo test app::tests::app_passes_rt_settings_to_editor_ui_frame_state --lib
Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
```

Expected: both focused tests pass.

## Task 3: Add Render, Top Bar, And Console RT Controls

**Files:**
- Modify: `src/editor/ui.rs`

- [ ] **Step 1: Write failing source contract tests**

Add these tests to `src/editor/ui.rs`:

```rust
#[test]
fn render_panel_exposes_backend_selection_and_rt_temporal_controls() {
    let source = crate::render::source_checks::read_source("src/editor/ui.rs");
    let render_panel = source
        .split("fn show_render_panel")
        .nth(1)
        .expect("render panel should exist")
        .split("fn show_restir_panel")
        .next()
        .expect("render panel should end before sampling panel");

    for token in [
        "render_mode_combo(ui, &mut lighting.render_mode)",
        "RT Temporal",
        "rt.temporal_denoise_enabled",
        "rt.history_length",
        "rt.normal_threshold",
        "rt.depth_threshold",
        "sanitize_rt_temporal_thresholds(rt)",
    ] {
        assert!(render_panel.contains(token), "render panel missing {token}");
    }
}

#[test]
fn top_bar_and_console_report_requested_backend_and_rt_state() {
    let source = crate::render::source_checks::read_source("src/editor/ui.rs");

    let top_bar = source
        .split("fn show_top_bar")
        .nth(1)
        .expect("top bar should exist")
        .split("fn show_left_rail")
        .next()
        .expect("top bar should end before left rail");
    assert!(top_bar.contains("render_mode_label(lighting.render_mode)"));
    assert!(!top_bar.contains("\"VPT Editor\""));

    let console = source
        .split("fn show_console")
        .nth(1)
        .expect("console should exist")
        .split("fn show_viewport_overlay")
        .next()
        .expect("console should end before overlay");
    for token in [
        "render_mode={}",
        "rt_di={}",
        "rt_gi={}",
        "rt_temporal={}",
    ] {
        assert!(console.contains(token), "console missing {token}");
    }
}
```

- [ ] **Step 2: Run focused tests and confirm they fail**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'
cargo test editor::ui::tests::render_panel_exposes_backend_selection_and_rt_temporal_controls --lib
cargo test editor::ui::tests::top_bar_and_console_report_requested_backend_and_rt_state --lib
Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
```

Expected: assertion or compile failures because RT UI controls are not present.

- [ ] **Step 3: Implement render mode labels, combo, top bar, console, and RT temporal controls**

Update imports:

```rust
use crate::render::scene_ubo::{
    LightingDebugView, LightingSettings, RenderMode, VptDebugView, VptDenoiserMode,
};
```

Add helpers:

```rust
fn render_mode_combo(ui: &mut egui::Ui, mode: &mut RenderMode) {
    ui.selectable_value(mode, RenderMode::Auto, "Auto");
    ui.selectable_value(mode, RenderMode::Vpt, "VPT");
    ui.selectable_value(mode, RenderMode::Rt, "RT");
}

fn render_mode_label(mode: RenderMode) -> &'static str {
    match mode {
        RenderMode::Auto => "Auto",
        RenderMode::Vpt => "VPT",
        RenderMode::Rt => "RT",
    }
}
```

Change `show_top_bar` to receive `rt: &RtSettings`, replace `"VPT Editor"` with the requested renderer label, and keep the denoiser combo.

Change `show_console` to receive `rt: &RtSettings` and add the RT summary fields.

Change `show_render_panel` signature to:

```rust
fn show_render_panel(ui: &mut egui::Ui, lighting: &mut LightingSettings, rt: &mut RtSettings)
```

Add the Renderer and RT Temporal sections while keeping existing VPT controls.

- [ ] **Step 4: Run focused tests and confirm they pass**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'
cargo test editor::ui::tests::render_panel_exposes_backend_selection_and_rt_temporal_controls --lib
cargo test editor::ui::tests::top_bar_and_console_report_requested_backend_and_rt_state --lib
Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
```

Expected: both focused tests pass.

## Task 4: Add RT Sampling And Debug Controls

**Files:**
- Modify: `src/editor/ui.rs`

- [ ] **Step 1: Write failing source contract tests**

Add these tests to `src/editor/ui.rs`:

```rust
#[test]
fn sampling_panel_exposes_rt_restir_controls() {
    let source = crate::render::source_checks::read_source("src/editor/ui.rs");
    let sampling_panel = source
        .split("fn show_restir_panel")
        .nth(1)
        .expect("sampling panel should exist")
        .split("fn show_debug_panel")
        .next()
        .expect("sampling panel should end before debug panel");

    for token in [
        "RT ReSTIR",
        "rt.restir_di_enabled",
        "rt.restir_di_spatial_enabled",
        "rt.restir_di_spatial_sample_count",
        "rt.restir_gi_enabled",
        "clamp_rt_spatial_samples(rt.restir_di_spatial_sample_count)",
    ] {
        assert!(sampling_panel.contains(token), "sampling panel missing {token}");
    }
}

#[test]
fn debug_panel_exposes_independent_rt_debug_controls() {
    let source = crate::render::source_checks::read_source("src/editor/ui.rs");
    let debug_panel = source
        .split("fn show_debug_panel")
        .nth(1)
        .expect("debug panel should exist")
        .split("fn denoiser_combo")
        .next()
        .expect("debug panel should end before combo helpers");

    for token in [
        "RT Debug",
        "debug_rt_view",
        "rt.debug_view",
        "rt_debug_combo(ui, &mut rt.debug_view)",
    ] {
        assert!(debug_panel.contains(token), "debug panel missing {token}");
    }

    assert!(
        !debug_panel.contains("set_rt_debug_view"),
        "RT debug should be independent direct RtSettings mutation in this phase"
    );
}
```

- [ ] **Step 2: Run focused tests and confirm they fail**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'
cargo test editor::ui::tests::sampling_panel_exposes_rt_restir_controls --lib
cargo test editor::ui::tests::debug_panel_exposes_independent_rt_debug_controls --lib
Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
```

Expected: assertion or compile failures because RT sampling/debug controls are not present.

- [ ] **Step 3: Implement RT ReSTIR and RT debug controls**

Change `show_restir_panel` signature to include `rt: &mut RtSettings`, then add:

```rust
ui.separator();
ui.label("RT ReSTIR");
ui.checkbox(&mut rt.restir_di_enabled, "Enable RT ReSTIR-DI");
ui.checkbox(&mut rt.restir_di_spatial_enabled, "RT DI spatial reuse");
ui.add(
    egui::Slider::new(&mut rt.restir_di_spatial_sample_count, 0..=8)
        .text("RT DI spatial samples"),
);
rt.restir_di_spatial_sample_count = clamp_rt_spatial_samples(rt.restir_di_spatial_sample_count);
ui.checkbox(&mut rt.restir_gi_enabled, "Enable RT ReSTIR-GI");
```

Change `show_debug_panel` signature to include `rt: &mut RtSettings`, then add:

```rust
ui.separator();
ui.label("RT Debug");
egui::ComboBox::from_id_salt("debug_rt_view")
    .selected_text(rt_debug_label(rt.debug_view))
    .show_ui(ui, |ui| {
        rt_debug_combo(ui, &mut rt.debug_view);
    });
```

Add:

```rust
fn rt_debug_combo(ui: &mut egui::Ui, debug_view: &mut RtDebugView) {
    for (view, label) in RT_DEBUG_OPTIONS {
        ui.selectable_value(debug_view, *view, *label);
    }
}
```

- [ ] **Step 4: Run focused tests and confirm they pass**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'
cargo test editor::ui::tests::sampling_panel_exposes_rt_restir_controls --lib
cargo test editor::ui::tests::debug_panel_exposes_independent_rt_debug_controls --lib
Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
```

Expected: both focused tests pass.

## Task 5: Final Verification

**Files:**
- Verify all modified files.

- [ ] **Step 1: Run formatting**

Run:

```powershell
cargo fmt --check
```

Expected: exit code 0. If it fails, run `cargo fmt`, inspect the diff, then rerun `cargo fmt --check`.

- [ ] **Step 2: Run the full library test gate**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'
cargo test --lib
Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
```

Expected: all library tests pass.

- [ ] **Step 3: Run diff hygiene**

Run:

```powershell
git diff --check
git status --short
```

Expected: no whitespace errors. `git status --short` should show only intended files until they are committed.

- [ ] **Step 4: Commit implementation**

Run:

```powershell
git add src/editor/ui.rs src/app.rs docs/superpowers/plans/2026-07-06-rt-ui-integration.md
git commit -m "feat: expose RT controls in editor UI"
```

Expected: implementation commit is created on `codex/rt-ui-integration`.
