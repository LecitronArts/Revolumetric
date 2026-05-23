# VPT NRD Motion Z Phase 2C Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move VPT `motion_history.z` from packed `motion_id` to NRD-style 2.5D depth delta while preserving `motion_id` in an independent guide image.

**Architecture:** Add current/previous `R32_UINT` `surface_motion_id` images beside the existing surface guides. The surface shader writes `hit.motion_id` to that guide and writes `previous_clip.w - view_z` into `motion_history.z` only when reprojection is valid; temporal consumers continue to use only the shared `motion.xy` reprojection contract.

**Tech Stack:** Rust render pass/resource graph code, Slang compute shaders, existing shader source contract tests, Vulkan storage images.

---

### Task 1: Contract Tests

**Files:**
- Modify: `src/render/passes/vpt/shader_source_tests.rs`

- [x] **Step 1: Write failing tests**

Add tests that assert:

```rust
assert!(surface.contains("RWTexture2D<uint> surface_motion_id"));
assert!(surface.contains("surface_motion_id[pixel] = hit.motion_id;"));
assert!(surface.contains("motion.z = previous_view_z - view_z;"));
assert!(!surface.contains("motion.z = float(hit.motion_id);"));
assert!(surface_rs.contains("pub surface_motion_id: GpuImage"));
assert!(surface_rs.contains("pub previous_surface_motion_id: GpuImage"));
assert!(surface_rs.contains("vk::Format::R32_UINT"));
assert!(compact_surface.contains("graph.bind_image(surface_writes.motion_id,self.surface_motion_id.handle);"));
assert!(compact_surface.contains("copy_surface_image(device,cmd,&self.surface_motion_id,&self.previous_surface_motion_id,);"));
```

- [x] **Step 2: Run tests to verify failure**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib vpt_motion_history_z_writes_view_z_delta_and_preserves_motion_id -- --exact
```

Expected: FAIL because production code still packs `hit.motion_id` into `motion.z` and does not declare `surface_motion_id`.

### Task 2: Surface Shader Guide

**Files:**
- Modify: `assets/shaders/passes/vpt_surface.slang`

- [x] **Step 1: Implement shader outputs**

Add `surface_motion_id` as `r32ui` storage image after `surface_view_z`, shift following bindings by one, write invalid id on miss, write `hit.motion_id` on hits, and set `motion.z` to `previous_view_z - view_z`.

- [x] **Step 2: Verify focused shader contract**

Run the focused test from Task 1. Expected: still FAIL until Rust resource/descriptor code is updated.

### Task 3: Rust Surface Resources

**Files:**
- Modify: `src/render/passes/vpt_surface.rs`

- [x] **Step 1: Add current/previous images**

Add `surface_motion_id` and `previous_surface_motion_id` to `VptSurfacePass`, `VptSurfaceImages`, and `VptSurfaceImageRefs`.

- [x] **Step 2: Add named graph fields**

Add `motion_id` to `VptCurrentSurfaceResources` and `VptPreviousSurfaceResources`; include it in graph writes, bind calls, and `for_each`.

- [x] **Step 3: Add descriptors and history copy**

Increase descriptor binding specs to include the new storage image, shift following bindings, include `surface_motion_id` in output image descriptors, allocate with `vk::Format::R32_UINT`, and copy current to previous during surface history update.

- [x] **Step 4: Verify focused tests**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib vpt_motion_history_z_writes_view_z_delta_and_preserves_motion_id vpt_surface_shader_binding_manifest_matches_expected_resources vpt_svgf_descriptor_specs_match_shader_manifests
```

Expected: PASS.

### Task 4: Regression Sweep

**Files:**
- Modify only files from Tasks 1-3 unless tests reveal a direct contract mismatch.

- [x] **Step 1: Run full verification**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo clippy --all-targets -- -D warnings
git diff --check
```

Expected: all exit 0.

- [x] **Step 2: Residual scan**

Run:

```powershell
rg -n "motion\.z = float\(hit\.motion_id\)|surface_motion_id\[" assets/shaders src/render
```

Expected: no old `motion.z = float(hit.motion_id)` token; `surface_motion_id[...]` appears only in the new guide writes/tests.
