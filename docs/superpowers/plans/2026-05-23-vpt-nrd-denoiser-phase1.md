# VPT NRD Denoiser Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the first executable NRD denoiser interface layer so `REVOLUMETRIC_DENOISER=off|svgf|relax|reblur` is parsed, recorded, and routed without changing the current GPU ABI or claiming an unavailable NRD backend.

**Architecture:** Phase 1 separates requested denoiser mode from effective runtime mode. `relax` and `reblur` are accepted as requested modes but resolve to the existing SVGF path until an NRD backend is implemented behind an explicit feature gate. The scene uniform ABI remains 224 bytes and the current SVGF temporal plus A-trous chain remains the only executable denoiser path.

**Tech Stack:** Rust, Vulkan/ash, Slang shaders, Cargo tests with `REVOLUMETRIC_SHADER_COMPILE=skip`.

---

## Scope

This plan implements only the low-risk interface and metadata layer from `docs/superpowers/specs/2026-05-23-vpt-nrd-reblur-relax-design.md`.

In scope:
- `REVOLUMETRIC_DENOISER` accepts `off`, `svgf`, `relax`, `reblur`, and existing boolean aliases.
- `LightingSettings` stores the requested mode as an enum.
- GPU `denoiser_flags` remains compatible with the existing SVGF shaders.
- `relax` and `reblur` explicitly fall back to SVGF through an `effective_denoiser_mode()` helper.
- Scene key changes when the requested denoiser mode changes.
- Capture metadata records requested mode, effective mode, and the existing boolean compatibility field.

Out of scope:
- NRD SDK vendoring or build integration.
- ReLAX/ReBLUR Vulkan resource creation.
- Diffuse/specular signal split.
- Hit-distance packing.
- Shader resource remapping.

## File Structure

- Modify `src/render/scene_ubo.rs`
  - Owns `VptDenoiserMode`, env parsing, effective-mode fallback policy, GPU flags, and ABI tests.
- Modify `src/render/passes/vpt_atrous.rs`
  - Uses `LightingSettings::denoiser_enabled()` instead of a boolean field.
- Modify `src/render/vpt_pipeline.rs`
  - Uses the denoiser mode discriminant in the scene key.
  - Passes requested and effective mode names into capture metadata.
- Modify `src/render/capture.rs`
  - Adds `denoiser_mode` and `effective_denoiser_mode` JSON fields while preserving `denoiser_enabled`.
- Do not modify `assets/shaders/**` in Phase 1.

## Task 1: Add Requested Denoiser Mode Parsing

**Files:**
- Modify: `src/render/scene_ubo.rs`

- [ ] **Step 1: Write failing parser test**

Add this test inside `#[cfg(test)] mod tests` in `src/render/scene_ubo.rs`:

```rust
#[test]
fn lighting_settings_parse_vpt_denoiser_modes_and_bool_aliases() {
    let cases = [
        ("off", VptDenoiserMode::Off),
        ("false", VptDenoiserMode::Off),
        ("0", VptDenoiserMode::Off),
        ("svgf", VptDenoiserMode::Svgf),
        ("on", VptDenoiserMode::Svgf),
        ("true", VptDenoiserMode::Svgf),
        ("1", VptDenoiserMode::Svgf),
        ("relax", VptDenoiserMode::Relax),
        ("reblur", VptDenoiserMode::Reblur),
    ];

    for (raw, expected) in cases {
        let result = LightingSettings::from_values_report_with_denoiser(
            None,
            None,
            None,
            None,
            None,
            None,
            Some(raw),
            None,
            None,
            None,
        );

        assert!(
            result.warnings.is_empty(),
            "denoiser mode {raw} should parse without warnings"
        );
        assert_eq!(result.settings.denoiser_mode, expected);
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib lighting_settings_parse_vpt_denoiser_modes_and_bool_aliases; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
```

Expected: FAIL to compile because `VptDenoiserMode` and `LightingSettings::denoiser_mode` do not exist.

- [ ] **Step 3: Implement requested mode enum and parser**

In `src/render/scene_ubo.rs`, add this enum near `VptDebugView`:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VptDenoiserMode {
    Off,
    Svgf,
    Relax,
    Reblur,
}
```

Add this impl near the other enum impl blocks:

```rust
impl VptDenoiserMode {
    pub fn as_config_value(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Svgf => "svgf",
            Self::Relax => "relax",
            Self::Reblur => "reblur",
        }
    }

    pub fn as_scene_key_value(self) -> u32 {
        match self {
            Self::Off => 0,
            Self::Svgf => 1,
            Self::Relax => 2,
            Self::Reblur => 3,
        }
    }
}
```

Change `LightingSettings`:

```rust
pub denoiser_mode: VptDenoiserMode,
```

Change its default:

```rust
denoiser_mode: VptDenoiserMode::Svgf,
```

Replace the `REVOLUMETRIC_DENOISER` override block with:

```rust
apply_optional_override(
    &mut settings.denoiser_mode,
    denoiser,
    "REVOLUMETRIC_DENOISER",
    "off|svgf|relax|reblur|on|1|true|false|0",
    parse_vpt_denoiser_mode,
    &mut warnings,
);
```

Add parser:

```rust
fn parse_vpt_denoiser_mode(value: &str) -> Option<VptDenoiserMode> {
    let value = value.trim();
    if value.eq_ignore_ascii_case("off")
        || value.eq_ignore_ascii_case("false")
        || value == "0"
    {
        Some(VptDenoiserMode::Off)
    } else if value.eq_ignore_ascii_case("svgf")
        || value.eq_ignore_ascii_case("on")
        || value.eq_ignore_ascii_case("true")
        || value == "1"
    {
        Some(VptDenoiserMode::Svgf)
    } else if value.eq_ignore_ascii_case("relax") {
        Some(VptDenoiserMode::Relax)
    } else if value.eq_ignore_ascii_case("reblur") {
        Some(VptDenoiserMode::Reblur)
    } else {
        None
    }
}
```

Update existing `LightingSettings` literals in `src/render/scene_ubo.rs` tests from:

```rust
denoiser_enabled: true,
```

to:

```rust
denoiser_mode: VptDenoiserMode::Svgf,
```

Update the existing off-denoiser assertion from:

```rust
assert!(!settings.denoiser_enabled);
```

to:

```rust
assert_eq!(settings.denoiser_mode, VptDenoiserMode::Off);
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib lighting_settings_parse_vpt_denoiser_modes_and_bool_aliases; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
```

Expected: PASS.

- [ ] **Step 5: Commit parser work**

Run:

```powershell
git add src/render/scene_ubo.rs
git commit -m "feat: add VPT denoiser mode parsing"
```

## Task 2: Preserve SVGF Fallback and Scene UBO ABI

**Files:**
- Modify: `src/render/scene_ubo.rs`

- [ ] **Step 1: Write failing effective-mode and ABI tests**

Add these tests in `src/render/scene_ubo.rs`:

```rust
#[test]
fn nrd_requested_modes_fall_back_to_svgf_until_backend_exists() {
    for requested in [VptDenoiserMode::Relax, VptDenoiserMode::Reblur] {
        let settings = LightingSettings {
            denoiser_mode: requested,
            ..LightingSettings::default()
        };

        assert_eq!(settings.effective_denoiser_mode(), VptDenoiserMode::Svgf);
        assert!(settings.denoiser_enabled());
        assert_eq!(settings.denoiser_flags(), DENOISER_FLAG_ENABLED);
    }
}

#[test]
fn denoiser_flags_follow_effective_mode_without_growing_scene_ubo() {
    let off = LightingSettings {
        denoiser_mode: VptDenoiserMode::Off,
        ..LightingSettings::default()
    };
    let svgf = LightingSettings {
        denoiser_mode: VptDenoiserMode::Svgf,
        ..LightingSettings::default()
    };

    assert_eq!(std::mem::size_of::<GpuSceneUniforms>(), 224);
    assert_eq!(off.denoiser_flags() & DENOISER_FLAG_ENABLED, 0);
    assert_eq!(svgf.denoiser_flags() & DENOISER_FLAG_ENABLED, DENOISER_FLAG_ENABLED);
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib nrd_requested_modes_fall_back_to_svgf_until_backend_exists denoiser_flags_follow_effective_mode_without_growing_scene_ubo; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
```

Expected: FAIL to compile because `effective_denoiser_mode()` and `denoiser_enabled()` do not exist.

- [ ] **Step 3: Implement effective-mode helpers**

Add methods to `impl LightingSettings` before `gpu_flags()`:

```rust
pub fn effective_denoiser_mode(self) -> VptDenoiserMode {
    match self.denoiser_mode {
        VptDenoiserMode::Off => VptDenoiserMode::Off,
        VptDenoiserMode::Svgf | VptDenoiserMode::Relax | VptDenoiserMode::Reblur => {
            VptDenoiserMode::Svgf
        }
    }
}

pub fn denoiser_enabled(self) -> bool {
    self.effective_denoiser_mode() != VptDenoiserMode::Off
}

pub fn denoiser_mode_name(self) -> &'static str {
    self.denoiser_mode.as_config_value()
}

pub fn effective_denoiser_mode_name(self) -> &'static str {
    self.effective_denoiser_mode().as_config_value()
}
```

Change `denoiser_flags()` to:

```rust
pub fn denoiser_flags(self) -> u32 {
    if self.denoiser_enabled() {
        DENOISER_FLAG_ENABLED
    } else {
        0
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib nrd_requested_modes_fall_back_to_svgf_until_backend_exists denoiser_flags_follow_effective_mode_without_growing_scene_ubo; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
```

Expected: PASS.

- [ ] **Step 5: Commit fallback and ABI work**

Run:

```powershell
git add src/render/scene_ubo.rs
git commit -m "feat: preserve SVGF fallback for NRD denoiser modes"
```

## Task 3: Route Effective Mode Through A-trous and Requested Mode Through Scene Key

**Files:**
- Modify: `src/render/passes/vpt_atrous.rs`
- Modify: `src/render/vpt_pipeline.rs`

- [ ] **Step 1: Write failing A-trous test**

In `src/render/passes/vpt_atrous.rs`, update `active_iteration_count_preserves_debug_and_disabled_passthrough` to use mode values:

```rust
#[test]
fn active_iteration_count_preserves_debug_and_disabled_passthrough() {
    let mut settings = LightingSettings::default();
    assert_eq!(VptAtrousPass::active_iteration_count(settings), 4);

    settings.denoiser_mode = crate::render::scene_ubo::VptDenoiserMode::Off;
    assert_eq!(VptAtrousPass::active_iteration_count(settings), 0);

    settings.denoiser_mode = crate::render::scene_ubo::VptDenoiserMode::Svgf;
    settings.vpt_debug_view = VptDebugView::Raw;
    assert_eq!(VptAtrousPass::active_iteration_count(settings), 0);

    settings.vpt_debug_view = VptDebugView::Final;
    settings.denoiser_atrous_iterations = 9;
    assert_eq!(
        VptAtrousPass::active_iteration_count(settings),
        MAX_ATROUS_ITERATIONS
    );
}
```

Add this new test:

```rust
#[test]
fn active_iteration_count_uses_svgf_fallback_for_nrd_modes() {
    for denoiser_mode in [
        crate::render::scene_ubo::VptDenoiserMode::Relax,
        crate::render::scene_ubo::VptDenoiserMode::Reblur,
    ] {
        let settings = LightingSettings {
            denoiser_mode,
            ..LightingSettings::default()
        };

        assert_eq!(VptAtrousPass::active_iteration_count(settings), 4);
    }
}
```

- [ ] **Step 2: Write failing scene-key test**

In `src/render/vpt_pipeline.rs`, update the imports in `#[cfg(test)] mod tests`:

```rust
use crate::render::scene_ubo::{
    LightingDebugView, LightingSettings, RenderMode, VptDebugView, VptDenoiserMode,
};
```

Add this test:

```rust
#[test]
fn scene_key_tracks_requested_denoiser_mode() {
    let base_settings = LightingSettings {
        denoiser_mode: VptDenoiserMode::Svgf,
        ..LightingSettings::default()
    };
    let relax_settings = LightingSettings {
        denoiser_mode: VptDenoiserMode::Relax,
        ..LightingSettings::default()
    };

    let base = VptRuntimePipeline::make_scene_key(
        glam::Vec3::new(0.5, 1.0, 0.25).normalize(),
        glam::Vec3::new(2.0, 1.5, 1.25),
        base_settings,
        false,
        false,
    );
    let relax = VptRuntimePipeline::make_scene_key(
        glam::Vec3::new(0.5, 1.0, 0.25).normalize(),
        glam::Vec3::new(2.0, 1.5, 1.25),
        relax_settings,
        false,
        false,
    );

    assert_ne!(base, relax);
}
```

- [ ] **Step 3: Run tests to verify they fail**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib active_iteration_count_uses_svgf_fallback_for_nrd_modes scene_key_tracks_requested_denoiser_mode; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
```

Expected: at least one test FAILS because `vpt_atrous.rs` still reads the old boolean field and `make_scene_key()` does not yet use the mode discriminant.

- [ ] **Step 4: Implement A-trous and scene-key routing**

Change `VptAtrousPass::active_iteration_count()`:

```rust
pub fn active_iteration_count(settings: LightingSettings) -> u32 {
    if settings.denoiser_enabled() && settings.vpt_debug_view == VptDebugView::Final {
        settings
            .denoiser_atrous_iterations
            .min(MAX_ATROUS_ITERATIONS)
    } else {
        0
    }
}
```

Change the denoiser slot in `VptRuntimePipeline::make_scene_key()` from:

```rust
lighting_settings.denoiser_enabled as u32,
```

to:

```rust
lighting_settings.denoiser_mode.as_scene_key_value(),
```

Update `LightingSettings` literals in `src/render/vpt_pipeline.rs` tests from:

```rust
denoiser_enabled: true,
```

to:

```rust
denoiser_mode: VptDenoiserMode::Svgf,
```

and from:

```rust
denoiser_enabled: false,
```

to:

```rust
denoiser_mode: VptDenoiserMode::Off,
```

- [ ] **Step 5: Run tests to verify they pass**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib active_iteration_count_uses_svgf_fallback_for_nrd_modes scene_key_tracks_requested_denoiser_mode; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
```

Expected: PASS.

- [ ] **Step 6: Commit routing work**

Run:

```powershell
git add src/render/passes/vpt_atrous.rs src/render/vpt_pipeline.rs
git commit -m "feat: route VPT denoiser mode through runtime keys"
```

## Task 4: Record Requested and Effective Denoiser Modes in Captures

**Files:**
- Modify: `src/render/capture.rs`
- Modify: `src/render/vpt_pipeline.rs`

- [ ] **Step 1: Write failing capture metadata test**

In `src/render/capture.rs`, update the `CaptureMetadata` literal in `metadata_json_records_frame_settings_and_paths`:

```rust
denoiser_enabled: true,
denoiser_mode: "relax",
effective_denoiser_mode: "svgf",
```

Add assertions:

```rust
assert!(json.contains("\"denoiser_enabled\": true"));
assert!(json.contains("\"denoiser_mode\": \"relax\""));
assert!(json.contains("\"effective_denoiser_mode\": \"svgf\""));
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib metadata_json_records_frame_settings_and_paths; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
```

Expected: FAIL to compile because the new metadata fields do not exist.

- [ ] **Step 3: Implement capture metadata fields**

Add fields to `CaptureMetadata` after `vpt_debug_view`:

```rust
pub denoiser_enabled: bool,
pub denoiser_mode: &'static str,
pub effective_denoiser_mode: &'static str,
```

Change the JSON format tail from:

```rust
"  \"vpt_debug_view\": \"{}\",\n",
"  \"denoiser_enabled\": {}\n",
```

to:

```rust
"  \"vpt_debug_view\": \"{}\",\n",
"  \"denoiser_enabled\": {},\n",
"  \"denoiser_mode\": \"{}\",\n",
"  \"effective_denoiser_mode\": \"{}\"\n",
```

Append arguments to `format!()`:

```rust
self.denoiser_enabled,
json_escape(self.denoiser_mode),
json_escape(self.effective_denoiser_mode)
```

In `src/render/vpt_pipeline.rs`, change capture construction from:

```rust
denoiser_enabled: inputs.lighting_settings.denoiser_enabled,
```

to:

```rust
denoiser_enabled: inputs.lighting_settings.denoiser_enabled(),
denoiser_mode: inputs.lighting_settings.denoiser_mode_name(),
effective_denoiser_mode: inputs.lighting_settings.effective_denoiser_mode_name(),
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib metadata_json_records_frame_settings_and_paths; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
```

Expected: PASS.

- [ ] **Step 5: Commit capture metadata work**

Run:

```powershell
git add src/render/capture.rs src/render/vpt_pipeline.rs
git commit -m "feat: record VPT denoiser modes in captures"
```

## Task 5: Full Verification and Cleanup

**Files:**
- Check: `src/render/scene_ubo.rs`
- Check: `src/render/passes/vpt_atrous.rs`
- Check: `src/render/vpt_pipeline.rs`
- Check: `src/render/capture.rs`

- [ ] **Step 1: Verify no old field references remain**

Run:

```powershell
Select-String -Path 'src/**/*.rs' -Pattern 'denoiser_enabled' -CaseSensitive:$false
```

Expected:
- `src/render/capture.rs` still has the compatibility metadata field.
- Other Rust files use `denoiser_enabled()` as a method call or no longer mention the old field.
- No `LightingSettings { denoiser_enabled: ... }` struct literals remain.

- [ ] **Step 2: Format**

Run:

```powershell
cargo fmt
```

Expected: exit code 0.

- [ ] **Step 3: Run library tests**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
```

Expected: exit code 0.

- [ ] **Step 4: Run clippy**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo clippy --all-targets -- -D warnings; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
```

Expected: exit code 0.

- [ ] **Step 5: Check whitespace and staged scope**

Run:

```powershell
git diff --check
git status --short
```

Expected:
- `git diff --check` exit code 0.
- Only Phase 1 files are unstaged or staged by this work.
- Pre-existing unrelated dirty files remain unstaged.

- [ ] **Step 6: Commit verification cleanup if needed**

If formatting or cleanup changed files after Task 4, run:

```powershell
git add src/render/scene_ubo.rs src/render/passes/vpt_atrous.rs src/render/vpt_pipeline.rs src/render/capture.rs
git commit -m "chore: verify VPT denoiser mode integration"
```

If there are no changes, do not create an empty commit.

## Self-Review

Spec coverage:
- Covers the Phase 1 executable interface from the NRD design spec.
- Preserves default SVGF behavior.
- Accepts future NRD mode names without wiring unavailable SDK code.
- Keeps `GpuSceneUniforms` ABI size at 224 bytes.
- Records requested and effective mode in capture metadata.

Known gaps:
- No NRD backend exists after this plan. This is intentional for Phase 1 and is visible through `effective_denoiser_mode()`.
- No runtime warning is added in `src/app.rs` because that file currently has unrelated worktree changes. Capture metadata and tests still prevent silent metadata misrepresentation.

Placeholder scan:
- No placeholder markers.
- No incomplete task steps.
- Every code-changing step includes concrete code and a verification command.

Type consistency:
- `VptDenoiserMode` is the stored requested mode.
- `LightingSettings::effective_denoiser_mode()` is the executable mode.
- `LightingSettings::denoiser_enabled()` is the compatibility boolean.
- Capture metadata uses `denoiser_mode` for requested mode and `effective_denoiser_mode` for executable mode.
