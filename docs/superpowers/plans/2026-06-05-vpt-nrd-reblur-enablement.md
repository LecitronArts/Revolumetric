# VPT NRD ReBLUR Enablement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `REVOLUMETRIC_DENOISER=reblur` route through the official NRD ReBLUR path with correct hit-distance packing, backend settings, and fallback behavior.

**Architecture:** Keep the existing VPT surface and NRD guide passes. Extend the native NRD wrapper so it can create/configure ReBLUR, then teach the frontend pass to pack normalized hit distance from `surface_view_z`. Finally, route `Reblur` through the NRD adapter/resolve chain while preserving `Svgf` as the non-NRD fallback.

**Tech Stack:** Rust, C++17, Slang, Vulkan/ash, NVIDIA NRD SDK

---

### Task 1: Native and Rust NRD ABI for ReBLUR

**Files:**
- Modify: `native/nrd_adapter.h`
- Modify: `native/nrd_adapter.cpp`
- Modify: `src/render/nrd_sys.rs`
- Modify: `src/render/nrd_adapter.rs`
- Test: `src/render/nrd_adapter.rs` tests

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn nrd_instance_exposes_reblur_entrypoints_when_feature_is_enabled() {
    // assert ReBLUR-related functions and settings exist in the Rust ABI layer
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test nrd_instance_exposes_reblur_entrypoints_when_feature_is_enabled --lib`
Expected: FAIL because the ReBLUR ABI is not wired yet.

- [ ] **Step 3: Write minimal implementation**

Add `ReblurSettings` / `reblur_diffuse` entrypoints in the C ABI and Rust wrapper, plus the settings conversion needed for `hitDistanceParameters`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test nrd_instance_exposes_reblur_entrypoints_when_feature_is_enabled --lib`
Expected: PASS.

### Task 2: VPT NRD Frontend ReBLUR Packing

**Files:**
- Modify: `assets/shaders/passes/vpt_nrd_frontend.slang`
- Modify: `src/render/passes/vpt_nrd_frontend.rs`
- Test: shader source tests under `src/render/passes/`

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn vpt_nrd_frontend_reblur_packs_normalized_hit_distance_from_view_z() {
    // assert the shader references surface_view_z and REBLUR packing helpers
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test vpt_nrd_frontend_reblur_packs_normalized_hit_distance_from_view_z --lib`
Expected: FAIL because the shader still packs RELAX-style hit distance.

- [ ] **Step 3: Write minimal implementation**

Read `surface_view_z`, compute normalized hit distance with `REBLUR_FrontEnd_GetNormHitDist`, and pack diffuse/specular payloads with the ReBLUR helpers.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test vpt_nrd_frontend_reblur_packs_normalized_hit_distance_from_view_z --lib`
Expected: PASS.

### Task 3: Runtime Routing and Fallback Semantics

**Files:**
- Modify: `src/render/scene_ubo.rs`
- Modify: `src/render/vpt_pipeline.rs`
- Modify: `src/render/passes/vpt_nrd_adapter.rs`
- Modify: `src/render/passes/vpt_nrd_resolve.rs`
- Test: existing VPT NRD runtime tests

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn requested_reblur_no_longer_falls_back_to_svgf_when_nrd_is_available() {
    // assert the effective mode and routing preserve ReBLUR when the backend exists
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test requested_reblur_no_longer_falls_back_to_svgf_when_nrd_is_available --lib`
Expected: FAIL because `Reblur` still collapses to `Svgf` in runtime routing.

- [ ] **Step 3: Write minimal implementation**

Update `effective_denoiser_mode()`, route `Reblur` through the NRD adapter/resolve path, and keep explicit fallback to `Svgf` only when NRD is unavailable.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test requested_reblur_no_longer_falls_back_to_svgf_when_nrd_is_available --lib`
Expected: PASS.

### Task 4: Verification

**Files:**
- No code changes unless verification exposes a regression

- [ ] **Step 1: Run formatting and tests**

Run:
```powershell
cargo fmt
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
```

- [ ] **Step 2: Run targeted NRD checks**

Run:
```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='strict'; cargo test --lib
```

- [ ] **Step 3: Inspect diff**

Run: `git diff --check`
Expected: no whitespace or patch-format errors.
