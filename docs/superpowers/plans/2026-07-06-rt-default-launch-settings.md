# RT Default Launch Settings Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Default RT-capable startup should run the full RT ReSTIR-DI/spatial/GI profile while keeping `REVOLUMETRIC_RENDER_MODE=auto` and all explicit overrides intact.

**Architecture:** `RtSettings::default()` remains the single source of truth for startup, UI, invalid-env fallback, and GPU uniform defaults. README and source-check tests are updated so documented launch behavior matches code.

**Tech Stack:** Rust 2024, Vulkan/ash, Cargo tests, repository source-check tests.

---

### Task 1: Default RT Settings

**Files:**
- Modify: `src/render/rt_settings.rs`

- [ ] **Step 1: Write the failing test**

Update `rt_settings_defaults_keep_temporal_denoise_enabled_and_restir_disabled` into a new expectation:

```rust
#[test]
fn rt_settings_defaults_enable_full_rt_restir_startup_profile() {
    let settings = RtSettings::default();

    assert!(settings.restir_di_enabled);
    assert!(settings.restir_gi_enabled);
    assert!(settings.temporal_denoise_enabled);
    assert!(settings.restir_di_spatial_enabled);
    assert_eq!(settings.restir_di_spatial_sample_count, 4);
    assert_eq!(settings.history_length, 20);
    assert_eq!(settings.normal_threshold, 0.85);
    assert_eq!(settings.depth_threshold, 0.02);
    assert_eq!(settings.debug_view, RtDebugView::Off);
}
```

Also update `rt_gpu_uniforms_layout_is_stable` default flag assertions:

```rust
assert_eq!(uniforms.restir_di_enabled, 1);
assert_eq!(uniforms.temporal_denoise_enabled, 1);
assert_eq!(uniforms.restir_di_spatial_enabled, 1);
```

- [ ] **Step 2: Verify RED**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib rt_settings_defaults_enable_full_rt_restir_startup_profile rt_gpu_uniforms_layout_is_stable
```

Expected: default-setting assertions fail because RT ReSTIR flags are still `false`/`0`.

- [ ] **Step 3: Implement minimal default change**

Change only the three default fields:

```rust
restir_di_enabled: true,
restir_gi_enabled: true,
restir_di_spatial_enabled: true,
```

- [ ] **Step 4: Verify GREEN**

Run the same focused command. Expected: both tests pass.

### Task 2: Docs And Source Checks

**Files:**
- Modify: `README.md`
- Modify: `src/render/source_checks.rs`

- [ ] **Step 1: Write failing source-check expectations**

In `readme_documents_local_validation_and_native_reblur_status`, replace the explicit RT ReSTIR env-token requirement with tokens proving the default-on profile and simplified RT smoke command are documented:

```rust
"RT ReSTIR defaults are enabled on the hardware RT backend",
"Default is `on`.",
"cargo run --features desktop --bin revolumetric # explicit RT ReSTIR default smoke",
"default auto backend smoke",
```

- [ ] **Step 2: Verify RED**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib readme_documents_local_validation_and_native_reblur_status
```

Expected: README token assertions fail until docs are updated.

- [ ] **Step 3: Update README**

Document that RT ReSTIR-DI, RT ReSTIR-DI spatial reuse, and RT ReSTIR-GI default to `on`, and show the RT smoke command without manually setting those three env vars.

- [ ] **Step 4: Verify GREEN**

Run the same focused source-check command. Expected: pass.

### Task 3: Final Verification

**Files:**
- No additional planned edits.

- [ ] **Step 1: Format check**

Run:

```powershell
cargo fmt --check
```

- [ ] **Step 2: Library tests**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib
```

- [ ] **Step 3: Clippy**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo clippy --all-targets -- -D warnings
```

- [ ] **Step 4: Diff hygiene**

Run:

```powershell
git diff --check
git status --short
```

Expected: no whitespace errors; only intended docs/tests/default-setting files changed before commit.
